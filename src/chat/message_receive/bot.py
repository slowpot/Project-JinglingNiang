import traceback
import os
import re

from typing import Dict, Any, Optional
from maim_message import UserInfo, Seg

from src.common.logger import get_logger
from src.config.config import global_config
from src.mood.mood_manager import mood_manager  # 导入情绪管理器
from src.mood.advanced_mood_manager import advanced_mood_manager  # 导入高级情绪管理器
from src.chat.message_receive.chat_stream import get_chat_manager, ChatStream
from src.chat.message_receive.message import MessageRecv, MessageRecvS4U
from src.chat.message_receive.storage import MessageStorage
from src.chat.heart_flow.heartflow_message_processor import HeartFCMessageReceiver
from src.chat.utils.prompt_builder import Prompt, global_prompt_manager
from src.plugin_system.core import component_registry, events_manager, global_announcement_manager
from src.plugin_system.base import BaseCommand, EventType
from src.mais4u.mais4u_chat.s4u_msg_processor import S4UMessageProcessor
from src.person_info.person_info import Person

# 定义日志配置

# 获取项目根目录（假设本文件在src/chat/message_receive/下，根目录为上上上级目录）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# 配置主程序日志格式
logger = get_logger("chat")


def _check_ban_words(text: str, chat: ChatStream, userinfo: UserInfo) -> bool:
    """检查消息是否包含过滤词

    Args:
        text: 待检查的文本
        chat: 聊天对象
        userinfo: 用户信息

    Returns:
        bool: 是否包含过滤词
    """
    for word in global_config.message_receive.ban_words:
        if word in text:
            chat_name = chat.group_info.group_name if chat.group_info else "私聊"
            logger.info(f"[{chat_name}]{userinfo.user_nickname}:{text}")
            logger.info(f"[过滤词识别]消息中含有{word}，filtered")
            return True
    return False


def _check_ban_regex(text: str, chat: ChatStream, userinfo: UserInfo) -> bool:
    """检查消息是否匹配过滤正则表达式

    Args:
        text: 待检查的文本
        chat: 聊天对象
        userinfo: 用户信息

    Returns:
        bool: 是否匹配过滤正则
    """
    # 检查text是否为None或空字符串
    if text is None or not text:
        return False
    
    for pattern in global_config.message_receive.ban_msgs_regex:
        if re.search(pattern, text):
            chat_name = chat.group_info.group_name if chat.group_info else "私聊"
            logger.info(f"[{chat_name}]{userinfo.user_nickname}:{text}")
            logger.info(f"[正则表达式过滤]消息匹配到{pattern}，filtered")
            return True
    return False


class ChatBot:
    def __init__(self):
        self.bot = None  # bot 实例引用
        self._started = False
        self.mood_manager = mood_manager  # 获取情绪管理器单例
        self.advanced_mood_manager = advanced_mood_manager  # 获取高级情绪管理器单例
        self.heartflow_message_receiver = HeartFCMessageReceiver()  # 新增

        self.s4u_message_processor = S4UMessageProcessor()

    async def _ensure_started(self):
        """确保所有任务已启动"""
        if not self._started:
            logger.debug("确保ChatBot所有任务已启动")

            self._started = True

    async def _process_commands_with_new_system(self, message: MessageRecv):
        # sourcery skip: use-named-expression
        """使用新插件系统处理命令"""
        try:
            text = message.processed_plain_text

            # 使用新的组件注册中心查找命令
            command_result = component_registry.find_command_by_text(text)
            if command_result:
                command_class, matched_groups, command_info = command_result
                plugin_name = command_info.plugin_name
                command_name = command_info.name
                if (
                    message.chat_stream
                    and message.chat_stream.stream_id
                    and command_name
                    in global_announcement_manager.get_disabled_chat_commands(message.chat_stream.stream_id)
                ):
                    logger.info("用户禁用的命令，跳过处理")
                    return False, None, True

                message.is_command = True

                # 获取插件配置
                plugin_config = component_registry.get_plugin_config(plugin_name)

                # 创建命令实例
                command_instance: BaseCommand = command_class(message, plugin_config)
                command_instance.set_matched_groups(matched_groups)

                try:
                    # 执行命令
                    success, response, intercept_message = await command_instance.execute()

                    # 记录命令执行结果
                    if success:
                        logger.info(f"命令执行成功: {command_class.__name__} (拦截: {intercept_message})")
                    else:
                        logger.warning(f"命令执行失败: {command_class.__name__} - {response}")

                    # 根据命令的拦截设置决定是否继续处理消息
                    return True, response, not intercept_message  # 找到命令，根据intercept_message决定是否继续

                except Exception as e:
                    logger.error(f"执行命令时出错: {command_class.__name__} - {e}")
                    logger.error(traceback.format_exc())

                    try:
                        await command_instance.send_text(f"命令执行出错: {str(e)}")
                    except Exception as send_error:
                        logger.error(f"发送错误消息失败: {send_error}")

                    # 命令出错时，根据命令的拦截设置决定是否继续处理消息
                    return True, str(e), False  # 出错时继续处理消息

            # 没有找到命令，继续处理消息
            return False, None, True

        except Exception as e:
            logger.error(f"处理命令时出错: {e}")
            return False, None, True  # 出错时继续处理消息

    async def handle_notice_message(self, message: MessageRecv):
        if message.message_info.message_id == "notice":
            message.is_notify = True
            logger.info("notice消息")
            # print(message)

            return True

    async def do_s4u(self, message_data: Dict[str, Any]):
        message = MessageRecvS4U(message_data)
        group_info = message.message_info.group_info
        user_info = message.message_info.user_info

        get_chat_manager().register_message(message)
        chat = await get_chat_manager().get_or_create_stream(
            platform=message.message_info.platform,  # type: ignore
            user_info=user_info,  # type: ignore
            group_info=group_info,
        )

        message.update_chat_stream(chat)

        # 处理消息内容
        await message.process()

        _ = Person.register_person(
            platform=message.message_info.platform,  # type: ignore
            user_id=message.message_info.user_info.user_id,  # type: ignore
            nickname=user_info.user_nickname,  # type: ignore
        )

        await self.s4u_message_processor.process_message(message)

        return

    async def echo_message_process(self, raw_data: Dict[str, Any]) -> None:
        """
        用于专门处理回送消息ID的函数
        """
        message_data: Dict[str, Any] = raw_data.get("content", {})
        if not message_data:
            return
        message_type = message_data.get("type")
        if message_type != "echo":
            return
        mmc_message_id = message_data.get("echo")
        actual_message_id = message_data.get("actual_id")
        if MessageStorage.update_message(mmc_message_id, actual_message_id):
            logger.debug(f"更新消息ID成功: {mmc_message_id} -> {actual_message_id}")
        else:
            logger.warning(f"更新消息ID失败: {mmc_message_id} -> {actual_message_id}")

    async def message_process(self, message_data: Dict[str, Any]) -> None:
        """处理转化后的统一格式消息
        这个函数本质是预处理一些数据，根据配置信息和消息内容，预处理消息，并分发到合适的消息处理器中
        heart_flow模式：使用思维流系统进行回复
        - 包含思维流状态管理
        - 在回复前进行观察和状态更新
        - 回复后更新思维流状态
        - 消息过滤
        - 记忆激活
        - 意愿计算
        - 消息生成和发送
        - 表情包处理
        - 性能计时
        """
        try:
            # 确保所有任务已启动
            await self._ensure_started()

            platform = message_data["message_info"].get("platform")

            if platform == "amaidesu_default":
                await self.do_s4u(message_data)
                return

            if message_data["message_info"].get("group_info") is not None:
                message_data["message_info"]["group_info"]["group_id"] = str(
                    message_data["message_info"]["group_info"]["group_id"]
                )
            if message_data["message_info"].get("user_info") is not None:
                message_data["message_info"]["user_info"]["user_id"] = str(
                    message_data["message_info"]["user_info"]["user_id"]
                )
            # print(message_data)
            # logger.debug(str(message_data))
            message = MessageRecv(message_data)
            group_info = message.message_info.group_info
            user_info = message.message_info.user_info

            continue_flag, modified_message = await events_manager.handle_mai_events(
                EventType.ON_MESSAGE_PRE_PROCESS, message
            )
            if not continue_flag:
                return
            if modified_message and modified_message._modify_flags.modify_message_segments:
                message.message_segment = Seg(type="seglist", data=modified_message.message_segments)

            if await self.handle_notice_message(message):
                # return
                pass

            get_chat_manager().register_message(message)

            chat = await get_chat_manager().get_or_create_stream(
                platform=message.message_info.platform,  # type: ignore
                user_info=user_info,  # type: ignore
                group_info=group_info,
            )

            message.update_chat_stream(chat)

            # 处理消息内容，生成纯文本
            await message.process()

            # if await self.check_ban_content(message):
            #     logger.warning(f"检测到消息中含有违法，色情，暴力，反动，敏感内容，消息内容：{message.processed_plain_text}，发送者：{message.message_info.user_info.user_nickname}")
            #     return

            # 过滤检查
            if _check_ban_words(message.processed_plain_text, chat, user_info) or _check_ban_regex(  # type: ignore
                message.raw_message,  # type: ignore
                chat,
                user_info,  # type: ignore
            ):
                return

            # 命令处理 - 使用新插件系统检查并处理命令
            is_command, cmd_result, continue_process = await self._process_commands_with_new_system(message)

            # 如果是命令且不需要继续处理，则直接返回
            if is_command and not continue_process:
                await MessageStorage.store_message(message, chat)
                logger.info(f"命令处理完成，跳过后续消息处理: {cmd_result}")
                return

            continue_flag, modified_message = await events_manager.handle_mai_events(EventType.ON_MESSAGE, message)
            if not continue_flag:
                return
            if modified_message and modified_message._modify_flags.modify_plain_text:
                message.processed_plain_text = modified_message.plain_text

            # 确认从接口发来的message是否有自定义的prompt模板信息
            if message.message_info.template_info and not message.message_info.template_info.template_default:
                template_group_name: Optional[str] = message.message_info.template_info.template_name  # type: ignore
                template_items = message.message_info.template_info.template_items
                async with global_prompt_manager.async_message_scope(template_group_name):
                    if isinstance(template_items, dict):
                        for k in template_items.keys():
                            await Prompt.create_async(template_items[k], k)
                            logger.debug(f"注册{template_items[k]},{k}")
            else:
                template_group_name = None

            async def preprocess():
                await self.heartflow_message_receiver.process_message(message)

            if template_group_name:
                async with global_prompt_manager.async_message_scope(template_group_name):
                    await preprocess()
            else:
                await preprocess()

        except Exception as e:
            logger.error(f"预处理消息失败: {e}")
            traceback.print_exc()


# 创建全局ChatBot实例
chat_bot = ChatBot()
