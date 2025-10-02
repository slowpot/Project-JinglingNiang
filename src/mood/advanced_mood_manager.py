import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from src.common.logger import get_logger
from src.config.config import global_config, model_config
from src.chat.message_receive.message import MessageRecv
from src.chat.message_receive.chat_stream import get_chat_manager
from src.chat.utils.prompt_builder import Prompt, global_prompt_manager
from src.chat.utils.chat_message_builder import build_readable_messages, get_raw_msg_by_timestamp_with_chat_inclusive
from src.llm_models.utils_model import LLMRequest
from src.manager.async_task_manager import AsyncTask, async_task_manager

logger = get_logger("advanced_mood")


class EmotionType(Enum):
    """基本情感类型"""
    JOY = "joy"           # 喜悦
    SADNESS = "sadness"   # 悲伤
    ANGER = "anger"       # 愤怒
    FEAR = "fear"         # 恐惧
    SURPRISE = "surprise" # 惊讶
    DISGUST = "disgust"   # 厌恶


class PersonalityTrait(Enum):
    """个性特质维度"""
    EXTROVERSION = "extroversion"    # 外向性
    NEUROTICISM = "neuroticism"      # 神经质
    OPENNESS = "openness"            # 开放性
    AGREEABLENESS = "agreeableness"  # 宜人性
    CONSCIENTIOUSNESS = "conscientiousness"  # 尽责性


@dataclass
class EmotionState:
    """情感状态数据类"""
    # 基本情感维度 (0-10分)
    joy: float = 5.0          # 喜悦
    sadness: float = 1.0       # 悲伤
    anger: float = 1.0         # 愤怒
    fear: float = 1.0          # 恐惧
    surprise: float = 1.0      # 惊讶
    disgust: float = 1.0       # 厌恶
    
    # 情感强度维度 (0-10分)
    arousal: float = 5.0       # 唤醒度 (平静-兴奋)
    valence: float = 5.0       # 效价 (负面-正面)
    dominance: float = 5.0     # 支配度 (被动-主动)
    
    # 复合情感维度 (0-10分)
    excitement: float = 5.0    # 兴奋度
    calmness: float = 5.0      # 平静度
    tension: float = 1.0       # 紧张度
    
    # 个性特质 (0-10分)
    extroversion: float = 5.0  # 外向性
    neuroticism: float = 5.0   # 神经质
    openness: float = 5.0      # 开放性
    agreeableness: float = 5.0 # 宜人性
    conscientiousness: float = 5.0  # 尽责性
    
    def normalize(self):
        """规范化情感值到0-10范围"""
        for attr in self.__dict__:
            if hasattr(self, attr):
                value = getattr(self, attr)
                setattr(self, attr, max(0.0, min(10.0, value)))
    
    def get_primary_emotion(self) -> Tuple[EmotionType, float]:
        """获取主导情感类型和强度"""
        emotions = {
            EmotionType.JOY: self.joy,
            EmotionType.SADNESS: self.sadness,
            EmotionType.ANGER: self.anger,
            EmotionType.FEAR: self.fear,
            EmotionType.SURPRISE: self.surprise,
            EmotionType.DISGUST: self.disgust
        }
        primary_emotion = max(emotions.items(), key=lambda x: x[1])
        return primary_emotion
    
    def get_emotion_description(self) -> str:
        """生成情感状态描述"""
        primary_emotion, intensity = self.get_primary_emotion()
        
        # 根据强度分级
        if intensity < 3:
            intensity_desc = "轻微"
        elif intensity < 6:
            intensity_desc = "中等"
        elif intensity < 8:
            intensity_desc = "强烈"
        else:
            intensity_desc = "非常强烈"
        
        # 根据情感类型生成描述
        emotion_descriptions = {
            EmotionType.JOY: f"{intensity_desc}的喜悦",
            EmotionType.SADNESS: f"{intensity_desc}的悲伤",
            EmotionType.ANGER: f"{intensity_desc}的愤怒",
            EmotionType.FEAR: f"{intensity_desc}的恐惧",
            EmotionType.SURPRISE: f"{intensity_desc}的惊讶",
            EmotionType.DISGUST: f"{intensity_desc}的厌恶"
        }
        
        # 添加情感强度描述
        if self.arousal > 7:
            arousal_desc = "兴奋的"
        elif self.arousal < 3:
            arousal_desc = "平静的"
        else:
            arousal_desc = ""
        
        description = emotion_descriptions[primary_emotion]
        if arousal_desc:
            description = f"{arousal_desc}{description}"
            
        return description
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式"""
        return {attr: getattr(self, attr) for attr in self.__dict__}
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'EmotionState':
        """从字典创建情感状态"""
        state = cls()
        for attr in state.__dict__:
            if attr in data:
                setattr(state, attr, data[attr])
        state.normalize()
        return state


@dataclass
class EmotionInfluence:
    """情感影响因子"""
    # 基本情感影响
    joy_influence: float = 0.0
    sadness_influence: float = 0.0
    anger_influence: float = 0.0
    fear_influence: float = 0.0
    surprise_influence: float = 0.0
    disgust_influence: float = 0.0
    
    # 强度维度影响
    arousal_influence: float = 0.0
    valence_influence: float = 0.0
    dominance_influence: float = 0.0


class LLMEmotionAnalyzer:
    """LLM情感分析器"""
    
    def __init__(self):
        self.emotion_model = LLMRequest(model_set=model_config.model_task_config.utils, request_type="emotion_analysis")
    
    async def analyze_emotion(self, text: str, context: str = "") -> EmotionInfluence:
        """使用LLM分析文本情感影响"""
        influence = EmotionInfluence()
        
        if not text:
            return influence
        
        # 构建情感分析prompt
        prompt = f"""
请分析以下文本的情感倾向，并给出对6种基本情感的影响程度（0-2分）：
- 喜悦(joy): 正面、愉快的情感
- 悲伤(sadness): 负面、难过的情感
- 愤怒(anger): 负面、生气的情绪
- 恐惧(fear): 负面、害怕的情绪
- 惊讶(surprise): 中性、意外的情绪
- 厌恶(disgust): 负面、反感的情绪

分析文本："{text}"

{context}

请以JSON格式输出分析结果，格式如下：
{{
    "joy": 0.0,
    "sadness": 0.0,
    "anger": 0.0,
    "fear": 0.0,
    "surprise": 0.0,
    "disgust": 0.0,
    "arousal": 0.0,
    "valence": 0.0,
    "dominance": 0.0
}}

说明：
- 每个情感维度的值范围是0.0到2.0
- arousal: 唤醒度（0=平静，2=兴奋）
- valence: 效价（0=负面，2=正面）
- dominance: 支配度（0=被动，2=主动）

请直接输出JSON，不要有其他内容。
"""
        
        try:
            response, (reasoning_content, _, _) = await self.emotion_model.generate_response_async(
                prompt=prompt, temperature=0.3, max_tokens=200
            )
            
            # 解析JSON响应
            import json
            emotion_data = json.loads(response.strip())
            
            # 映射到情感影响
            for emotion_type in EmotionType:
                if emotion_type.value in emotion_data:
                    setattr(influence, f"{emotion_type.value}_influence",
                           float(emotion_data[emotion_type.value]))
            
            # 设置强度维度
            if "arousal" in emotion_data:
                influence.arousal_influence = float(emotion_data["arousal"])
            if "valence" in emotion_data:
                influence.valence_influence = float(emotion_data["valence"])
            if "dominance" in emotion_data:
                influence.dominance_influence = float(emotion_data["dominance"])
                
        except Exception as e:
            logger.warning(f"LLM情感分析失败，使用备用方法: {e}")
            # 备用方法：使用简单的关键词分析
            influence = self._fallback_analysis(text)
        
        return influence
    
    def _fallback_analysis(self, text: str) -> EmotionInfluence:
        """备用情感分析方法（关键词匹配）"""
        influence = EmotionInfluence()
        
        emotion_keywords = {
            EmotionType.JOY: ["开心", "高兴", "快乐", "喜悦", "兴奋", "愉快", "欢乐", "哈哈", "嘻嘻"],
            EmotionType.SADNESS: ["悲伤", "难过", "伤心", "忧郁", "失落", "沮丧", "哭了", "泪目"],
            EmotionType.ANGER: ["生气", "愤怒", "恼火", "气愤", "暴躁", "不满", "讨厌", "烦人"],
            EmotionType.FEAR: ["害怕", "恐惧", "担心", "忧虑", "紧张", "恐慌", "吓死"],
            EmotionType.SURPRISE: ["惊讶", "惊奇", "意外", "震惊", "诧异", "居然", "竟然"],
            EmotionType.DISGUST: ["厌恶", "讨厌", "反感", "恶心", "嫌弃", "呕"]
        }
        
        # 关键词匹配分析
        for emotion_type, keywords in emotion_keywords.items():
            keyword_count = sum(1 for keyword in keywords if keyword in text)
            if keyword_count > 0:
                base_influence = min(1.5, keyword_count * 0.3)
                setattr(influence, f"{emotion_type.value}_influence", base_influence)
        
        # 消息长度影响唤醒度
        length_factor = min(1.0, len(text) / 100)
        influence.arousal_influence = length_factor * 0.5
        
        # 标点符号分析情感强度
        exclamation_count = text.count('!') + text.count('！')
        question_count = text.count('?') + text.count('？')
        
        if exclamation_count > 0:
            influence.arousal_influence += min(1.0, exclamation_count * 0.3)
        if question_count > 0:
            influence.surprise_influence += min(1.0, question_count * 0.2)
        
        return influence


class EmotionCalculator:
    """情感计算器"""
    
    def __init__(self):
        self.llm_analyzer = LLMEmotionAnalyzer()
    
    async def calculate_message_influence(self, message: MessageRecv, context_messages: list = None) -> EmotionInfluence:
        """计算消息对情感的影响（使用LLM分析）"""
        text = message.processed_plain_text or ""
        
        if not text:
            return EmotionInfluence()
        
        # 构建上下文
        context = ""
        if context_messages:
            context = f"上下文对话：{', '.join([msg.processed_plain_text or '' for msg in context_messages[-3:]])}"
        
        # 使用LLM进行情感分析
        influence = await self.llm_analyzer.analyze_emotion(text, context)
        
        return influence
    
    def apply_influence(self, current_state: EmotionState, influence: EmotionInfluence) -> EmotionState:
        """应用情感影响"""
        new_state = EmotionState()
        
        # 复制当前状态
        for attr in current_state.__dict__:
            setattr(new_state, attr, getattr(current_state, attr))
        
        # 应用基本情感影响
        for emotion_type in EmotionType:
            influence_attr = f"{emotion_type.value}_influence"
            current_value = getattr(new_state, emotion_type.value)
            influence_value = getattr(influence, influence_attr, 0.0)
            setattr(new_state, emotion_type.value, current_value + influence_value)
        
        # 应用强度维度影响
        new_state.arousal += influence.arousal_influence
        new_state.valence += influence.valence_influence
        new_state.dominance += influence.dominance_influence
        
        # 更新复合情感维度
        self._update_composite_emotions(new_state)
        
        new_state.normalize()
        return new_state
    
    def _update_composite_emotions(self, state: EmotionState):
        """更新复合情感维度"""
        # 兴奋度 = 唤醒度 + 喜悦 - 悲伤
        state.excitement = (state.arousal + state.joy - state.sadness) / 2
        
        # 平静度 = 10 - 唤醒度 - 紧张度
        state.calmness = 10 - state.arousal - state.tension
        
        # 紧张度 = 恐惧 + 愤怒 - 喜悦
        state.tension = (state.fear + state.anger - state.joy) / 2


class AdvancedChatMood:
    """高级聊天情绪管理"""
    
    def __init__(self, chat_id: str):
        self.chat_id: str = chat_id
        
        chat_manager = get_chat_manager()
        self.chat_stream = chat_manager.get_stream(self.chat_id)
        
        if not self.chat_stream:
            raise ValueError(f"Chat stream for chat_id {chat_id} not found")
        
        self.log_prefix = f"[{self.chat_stream.group_info.group_name if self.chat_stream.group_info else self.chat_stream.user_info.user_nickname}]"
        
        # 情感状态
        self.emotion_state = EmotionState()
        self.last_emotion_description: str = "感觉很平静"
        
        # 计算器
        self.calculator = EmotionCalculator()
        
        # 时间跟踪
        self.last_change_time: float = time.time()
        self.regression_count: int = 0
        
        # LLM模型
        self.mood_model = LLMRequest(model_set=model_config.model_task_config.utils, request_type="mood")
    
    async def update_mood_by_message(self, message: MessageRecv):
        """基于消息更新情感状态"""
        self.regression_count = 0
        
        # 计算更新概率
        update_probability = self._calculate_update_probability(message)
        if random.random() > update_probability:
            return
        
        logger.debug(f"{self.log_prefix} 更新情感状态，更新概率: {update_probability:.2f}")
        
        # 记录变化前的状态
        old_state = self.emotion_state.to_dict()
        old_description = self.last_emotion_description
        
        # 获取上下文消息
        message_time: float = message.message_info.time  # type: ignore
        context_messages = get_raw_msg_by_timestamp_with_chat_inclusive(
            chat_id=self.chat_id,
            timestamp_start=self.last_change_time,
            timestamp_end=message_time,
            limit=5,
            limit_mode="last",
        )
        
        # 计算情感影响（使用LLM分析）
        influence = await self.calculator.calculate_message_influence(message, context_messages)
        
        # 应用情感影响
        self.emotion_state = self.calculator.apply_influence(self.emotion_state, influence)
        
        # 使用LLM生成情感描述
        await self._generate_emotion_description(message)
        
        # 记录变化后的状态
        new_state = self.emotion_state.to_dict()
        new_description = self.last_emotion_description
        
        # 计算变化量
        state_changes = {}
        for key in old_state:
            if key in new_state:
                change = new_state[key] - old_state[key]
                if abs(change) > 0.01:  # 只记录显著变化
                    state_changes[key] = f"{old_state[key]:.1f} → {new_state[key]:.1f} (Δ{change:+.1f})"
        
        # 输出详细的情感变化日志
        logger.info(f"{self.log_prefix} === 情感状态变化 ===")
        logger.info(f"{self.log_prefix} 触发事件: 收到消息")
        logger.info(f"{self.log_prefix} 消息内容: '{message.processed_plain_text[:50]}{'...' if len(message.processed_plain_text) > 50 else ''}'")
        logger.info(f"{self.log_prefix} 时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(message_time))}")
        logger.info(f"{self.log_prefix} 情感描述变化: {old_description} → {new_description}")
        logger.info(f"{self.log_prefix} 更新概率: {update_probability:.2f}")
        logger.info(f"{self.log_prefix} 时间间隔: {during_last_time:.1f}秒")
        
        if state_changes:
            logger.info(f"{self.log_prefix} 情感维度变化:")
            # 按类别分组显示情感维度变化
            emotion_categories = {
                "基本情感": ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust'],
                "强度维度": ['arousal', 'valence', 'dominance'],
                "复合情感": ['excitement', 'calmness', 'tension'],
                "个性特质": ['extroversion', 'neuroticism', 'openness', 'agreeableness', 'conscientiousness']
            }
            
            for category, dimensions in emotion_categories.items():
                category_changes = {dim: state_changes[dim] for dim in dimensions if dim in state_changes}
                if category_changes:
                    logger.info(f"{self.log_prefix}   [{category}]")
                    for dimension, change in category_changes.items():
                        logger.info(f"{self.log_prefix}     {dimension}: {change}")
        
        # 记录情感影响详情
        influence_details = []
        for attr in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'arousal', 'valence', 'dominance']:
            influence_value = getattr(influence, f"{attr}_influence", 0.0)
            if abs(influence_value) > 0.01:
                influence_details.append(f"{attr}+{influence_value:+.1f}")
        
        if influence_details:
            logger.info(f"{self.log_prefix} 情感影响因子: {', '.join(influence_details)}")
        
        logger.info(f"{self.log_prefix} ===================")
        
        self.last_change_time = message_time
    
    def _calculate_update_probability(self, message: MessageRecv) -> float:
        """计算情感更新概率"""
        during_last_time = message.message_info.time - self.last_change_time  # type: ignore
        
        base_probability = 0.05
        time_multiplier = 4 * (1 - math.exp(-0.01 * during_last_time))
        
        # 基于消息长度计算兴趣度
        message_length = len(message.processed_plain_text or "")
        interest_multiplier = min(2.0, 1.0 + message_length / 100)
        
        update_probability = global_config.mood.mood_update_threshold * min(
            1.0, base_probability * time_multiplier * interest_multiplier
        )
        
        return update_probability
    
    async def _generate_emotion_description(self, message: MessageRecv):
        """使用LLM生成情感描述"""
        message_time: float = message.message_info.time  # type: ignore
        message_list = get_raw_msg_by_timestamp_with_chat_inclusive(
            chat_id=self.chat_id,
            timestamp_start=self.last_change_time,
            timestamp_end=message_time,
            limit=int(global_config.chat.max_context_size / 3),
            limit_mode="last",
        )
        
        chat_talking_prompt = build_readable_messages(
            message_list,
            replace_bot_name=True,
            timestamp_mode="normal_no_YMD",
            read_mark=0.0,
            truncate=True,
            show_actions=True,
        )
        
        bot_name = global_config.bot.nickname
        if global_config.bot.alias_names:
            bot_nickname = f",也有人叫你{','.join(global_config.bot.alias_names)}"
        else:
            bot_nickname = ""
        
        identity_block = f"你的名字是{bot_name}{bot_nickname}"
        
        # 构建包含情感数据的prompt
        emotion_data = self.emotion_state.to_dict()
        emotion_summary = ", ".join([f"{k}: {v:.1f}" for k, v in emotion_data.items()])
        
        prompt = f"""
{chat_talking_prompt}
以上是群里正在进行的聊天记录

{identity_block}
你当前的情感状态数据如下：{emotion_summary}

请基于当前聊天内容和你的情感数据，用一句话描述你现在的情绪状态。
你的情绪特点是:{global_config.personality.emotion_style}
请只输出情绪状态描述，不要输出其他内容：
"""
        
        response, (reasoning_content, _, _) = await self.mood_model.generate_response_async(
            prompt=prompt, temperature=0.7
        )
        
        if global_config.debug.show_prompt:
            logger.info(f"{self.log_prefix} prompt: {prompt}")
            logger.info(f"{self.log_prefix} response: {response}")
            logger.info(f"{self.log_prefix} reasoning_content: {reasoning_content}")
        
        self.last_emotion_description = response
    
    async def regress_mood(self):
        """情感回归（自然衰减）"""
        # 记录变化前的状态
        old_state = self.emotion_state.to_dict()
        old_description = self.last_emotion_description
        
        # 情感自然衰减
        decay_factor = 0.9  # 每次回归衰减10%
        for attr in self.emotion_state.__dict__:
            if hasattr(self.emotion_state, attr):
                current_value = getattr(self.emotion_state, attr)
                # 向中性值5.0衰减
                new_value = 5.0 + (current_value - 5.0) * decay_factor
                setattr(self.emotion_state, attr, new_value)
        
        self.emotion_state.normalize()
        
        # 更新情感描述
        self.last_emotion_description = self.emotion_state.get_emotion_description()
        
        # 记录变化后的状态
        new_state = self.emotion_state.to_dict()
        new_description = self.last_emotion_description
        
        # 计算变化量
        state_changes = {}
        for key in old_state:
            if key in new_state:
                change = new_state[key] - old_state[key]
                if abs(change) > 0.01:  # 只记录显著变化
                    state_changes[key] = f"{old_state[key]:.1f} → {new_state[key]:.1f} (Δ{change:+.1f})"
        
        # 输出详细的情感回归日志
        logger.info(f"{self.log_prefix} === 情感状态回归 ===")
        logger.info(f"{self.log_prefix} 触发事件: 自然衰减")
        logger.info(f"{self.log_prefix} 回归次数: 第{self.regression_count + 1}次")
        logger.info(f"{self.log_prefix} 时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
        logger.info(f"{self.log_prefix} 情感描述变化: {old_description} → {new_description}")
        logger.info(f"{self.log_prefix} 距离上次变化: {time.time() - self.last_change_time:.1f}秒")
        
        if state_changes:
            logger.info(f"{self.log_prefix} 情感维度变化:")
            # 按类别分组显示情感维度变化
            emotion_categories = {
                "基本情感": ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust'],
                "强度维度": ['arousal', 'valence', 'dominance'],
                "复合情感": ['excitement', 'calmness', 'tension'],
                "个性特质": ['extroversion', 'neuroticism', 'openness', 'agreeableness', 'conscientiousness']
            }
            
            for category, dimensions in emotion_categories.items():
                category_changes = {dim: state_changes[dim] for dim in dimensions if dim in state_changes}
                if category_changes:
                    logger.info(f"{self.log_prefix}   [{category}]")
                    for dimension, change in category_changes.items():
                        logger.info(f"{self.log_prefix}     {dimension}: {change}")
        
        logger.info(f"{self.log_prefix} 衰减因子: {decay_factor}")
        logger.info(f"{self.log_prefix} ===================")
        
        self.regression_count += 1
    
    def get_mood_description(self) -> str:
        """获取当前情感描述"""
        return self.last_emotion_description
    
    def get_emotion_data(self) -> Dict[str, float]:
        """获取情感数据"""
        return self.emotion_state.to_dict()


class AdvancedMoodRegressionTask(AsyncTask):
    """高级情绪回归任务"""
    
    def __init__(self, mood_manager: "AdvancedMoodManager"):
        super().__init__(task_name="AdvancedMoodRegressionTask", run_interval=45)
        self.mood_manager = mood_manager
    
    async def run(self):
        logger.debug("开始高级情绪回归任务...")
        now = time.time()
        for mood in self.mood_manager.mood_list:
            if mood.last_change_time == 0:
                continue
            
            if now - mood.last_change_time > 200:
                if mood.regression_count >= 3:  # 最多回归3次
                    continue
                
                logger.debug(f"{mood.log_prefix} 开始情感回归, 第 {mood.regression_count + 1} 次")
                await mood.regress_mood()


class AdvancedMoodManager:
    """高级情绪管理器"""
    
    def __init__(self):
        self.mood_list: list[AdvancedChatMood] = []
        self.task_started: bool = False
    
    async def start(self):
        """启动情绪回归后台任务"""
        if self.task_started:
            return
        
        logger.info("启动高级情绪回归任务...")
        task = AdvancedMoodRegressionTask(self)
        await async_task_manager.add_task(task)
        self.task_started = True
        logger.info("高级情绪回归任务已启动")
    
    def get_mood_by_chat_id(self, chat_id: str) -> AdvancedChatMood:
        """获取或创建聊天情绪状态"""
        for mood in self.mood_list:
            if mood.chat_id == chat_id:
                return mood
        
        new_mood = AdvancedChatMood(chat_id)
        self.mood_list.append(new_mood)
        return new_mood
    
    def reset_mood_by_chat_id(self, chat_id: str):
        """重置聊天情绪状态"""
        for mood in self.mood_list:
            if mood.chat_id == chat_id:
                mood.emotion_state = EmotionState()
                mood.last_emotion_description = "感觉很平静"
                mood.regression_count = 0
                return
        self.mood_list.append(AdvancedChatMood(chat_id))


# 全局高级情绪管理器
advanced_mood_manager = AdvancedMoodManager()