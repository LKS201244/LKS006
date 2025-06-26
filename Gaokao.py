'''
高考择校助手 - 智能聊天版
'''
from openai import OpenAI, Stream
import streamlit as st
from typing import Generator, Optional

from openai.types.chat import ChatCompletion, ChatCompletionChunk

# 全局配置
PROVINCES = ['北京', '天津', '河北', '山西', '内蒙古', '辽宁', '吉林', '黑龙江', '上海', '江苏', '浙江', '安徽', '福建', '江西', '山东', '河南', '湖北', '湖南', '广东', '广西', '海南', '重庆', '四川', '贵州', '云南', '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆', '香港', '澳门', '台湾']
SUBJECT_TYPES = ["文科", "理科", "物理类", "历史类", "新高考"]
INTEREST_EXAMPLES = "如：计算机、临床医学、金融学、法学、建筑学"

def get_llm_response(
    client: OpenAI,
    model: str,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    stream: bool = False
) -> ChatCompletion | Stream[ChatCompletionChunk]:
    """
    获取大语言模型响应

    参数:
        client: OpenAI客户端实例
        model: 使用的模型名称
        user_prompt: 用户提示词
        system_prompt: 系统提示词(可选)
        stream: 是否使用流式输出

    返回:
        生成器或完整响应
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    return client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream
    )

def get_advice(question: str, score: int, province: str, interests: str, subject_type: str) -> Generator[str, None, None]:
    """
    智能生成高考志愿建议（流式输出）

    参数:
        question: 用户咨询问题
        score: 高考分数(200-750)
        province: 所在省份
        interests: 兴趣专业
        subject_type: 考生科类

    返回:
        生成器逐块返回建议内容
    """
    try:
        # 专业级提示词模板
        prompt = f"""作为高考志愿填报专家，请根据以下考生信息提供专业建议：
        
考生档案：
- 所在省份：{province}（需考虑省内录取政策）
- 高考分数：{score}分（{_get_score_level(score)}）
- 报考科类：{subject_type}
- 兴趣方向：{interests}
- 咨询问题：{question}

请按以下框架回答：
1. 分数竞争力分析（省内排名预估）
2. 推荐院校清单（分冲刺/稳妥/保底三档）
3. 匹配专业推荐（结合兴趣和就业前景）
4. 志愿填报策略建议
5. 重要注意事项提醒"""

        client = OpenAI(base_url=base_url, api_key=api_key)
        stream = get_llm_response(
            client=client,
            model=model_name,
            user_prompt=prompt,
            system_prompt="你是一名资深高考志愿规划师，回答需专业准确",
            stream=True
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content or ''
            yield content

    except Exception as e:
        error_msg = f"⚠️ 服务暂时不可用（错误代码：{type(e).__name__}）"
        yield error_msg

def _get_score_level(score: int) -> str:
    """分数等级评估"""
    if score >= 650: return "顶尖水平（全省前1%）"
    elif score >= 600: return "优秀水平（全省前10%）"
    elif score >= 500: return "中等偏上"
    elif score >= 400: return "中等水平"
    return "一般水平"

# 页面配置
st.set_page_config(
    page_title="智能高考择校助手",
    page_icon="🏫",
    layout="centered"
)

# 侧边栏配置
with st.sidebar:
    st.header("🔧 系统配置")
    api_vendor = st.radio(
        "AI服务提供商",
        options=['OpenAI', 'deepseek'],
        index=0,
        horizontal=True
    )

    if api_vendor == 'OpenAI':
        base_url = st.selectbox(
            "API端点",
            options=['https://api.openai.com/v1', 'https://twapi.openai-hk.com/v1'],
            index=0
        )
        model_options = ['gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo']
    else:
        base_url = 'https://api.deepseek.com'
        model_options = ['deepseek-chat', 'deepseek-v2']

    model_name = st.selectbox("AI模型", options=model_options)
    api_key = st.text_input("API密钥", type="password", help="请输入有效的API访问密钥")

    st.divider()
    st.header("📋 考生档案")
    user_score = st.slider(
        "高考分数",
        min_value=200,
        max_value=750,
        value=500,
        step=1,
        help="请填写真实高考成绩"
    )
    user_province = st.selectbox("所在省份", options=PROVINCES, index=0)
    user_subject = st.radio("报考科类", options=SUBJECT_TYPES, index=0, horizontal=True)  # 修正变量名拼写
    user_interests = st.text_area(
        "兴趣专业方向",
        placeholder=INTEREST_EXAMPLES,
        help="可输入多个专业，用逗号分隔"
    )

# 主界面
st.title("🎯 智能高考择校助手")
st.caption("AI驱动的志愿填报咨询系统 | 数据仅供参考，请以官方发布为准")

# 初始化对话
if 'messages' not in st.session_state:
    welcome_msg = """您好！我是智能择校助手小高，请先完善侧边栏的：
1. 考生基本信息
2. AI服务配置

然后可以直接提问，例如：
- 我的分数能上哪些985大学？
- 推荐适合我的计算机专业院校
- 这个分数在省内的竞争力如何？"""
    st.session_state.messages = [{"role": "ai", "content": welcome_msg}]

# 显示对话历史
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 交互处理
if not api_key:
    st.warning("请先在侧边栏输入有效的API密钥")
    st.stop()

if user_input := st.chat_input("输入您的问题..."):
    # 验证必要信息
    if not all([user_province, user_subject, user_interests]):  # 使用修正后的变量名
        st.error("请先完善考生基本信息！")
        st.stop()

    # 添加用户消息
    st.session_state.messages.append({"role": "human", "content": user_input})
    st.chat_message("human").write(user_input)

    # 获取建议
    with st.status("🔍 正在分析数据，请稍候...", expanded=True) as status:
        try:
            advice_gen = get_advice(
                question=user_input,
                score=user_score,
                province=user_province,
                interests=user_interests,
                subject_type=user_subject  # 使用修正后的变量名
            )

            # 流式输出
            response = st.chat_message("ai").write_stream(advice_gen)
            st.session_state.messages.append({"role": "ai", "content": response})

            status.update(label="✅ 分析完成", state="complete")
        except Exception as e:
            st.error(f"系统错误: {str(e)}")