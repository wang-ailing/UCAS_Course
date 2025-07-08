# 姓氏库
surnames = set(["张", "李", "王", "赵", "刘", "陈", "杨", "黄", "吴", "周", "徐", "孙"])

# 名字字符统计（示例数据）
name_char_freq = {
    "建": {"freq": 100, "first_prob": 0.8, "last_prob": 0.2},
    "国": {"freq": 120, "first_prob": 0.4, "last_prob": 0.8},
    "庆": {"freq": 80, "first_prob": 0.2, "last_prob": 0.8},
    "江": {"freq": 60, "first_prob": 0.5, "last_prob": 0.5}
}

# 阈值表
surname_threshold = {"张": -3.0, "李": -3.2, "王": -3.1}

import math

def calculate_name_probability(surname, first_char, second_char=None):
    """计算姓名的概率"""
    if surname not in surnames:
        return -float('inf')  # 非姓氏直接排除
    
    # 获取名字的概率
    if first_char not in name_char_freq:
        return -float('inf')  # 名字首字未知
    
    p_surname = 1.0  # 假设姓氏概率固定为1
    p_first = name_char_freq[first_char]["first_prob"]
    
    if second_char:
        p_second = name_char_freq.get(second_char, {}).get("last_prob", 0.1)
        prob = p_surname * p_first * p_second
    else:
        prob = p_surname * p_first
    
    return math.log(prob + 1e-9)  # 防止 log(0)

def detect_names(text):
    """识别文本中的人名"""
    detected_names = []
    n = len(text)
    
    for i in range(n):
        # 检查是否为姓氏
        if text[i] in surnames:
            # 尝试一字名
            if i + 1 < n:
                prob = calculate_name_probability(text[i], text[i + 1])
                if prob > surname_threshold.get(text[i], -float('inf')):
                    detected_names.append(text[i:i+2])
            
            # 尝试二字名
            if i + 2 < n:
                prob = calculate_name_probability(text[i], text[i + 1], text[i + 2])
                if prob > surname_threshold.get(text[i], -float('inf')):
                    detected_names.append(text[i:i+3])
    return detected_names

def refine_names(text, names):
    """根据上下文规则修饰潜在人名"""
    refined = []
    for name in names:
        index = text.find(name)
        if index > 0:
            # 左边界规则
            if text[index - 1] in ["，", "。", "“", "”", "是"]:
                refined.append(name)
        if index + len(name) < len(text):
            # 右边界规则
            if text[index + len(name)] in ["，", "。", "“", "”", "说"]:
                refined.append(name)
    return list(set(refined))

if __name__ == '__main__':
    # Test case
    text_list = [
        "张建国正在演讲。",
        "李国庆发表了意见。",
        "南京市长江大桥是著名景点。"
    ]
    for text in text_list:
        detected_names = detect_names(text)
        print(detected_names)
        refined_names = refine_names(text, detected_names)
        print(refined_names)