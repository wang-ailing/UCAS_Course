import re
from collections import Counter
import math

word_dict = {
    "地铁", "草莓", "土豆", "幸福", "飞机", "布丁", "森林", "洗衣机", "蛋糕", "洋葱",
    "超市", "芒果", "西红柿", "牛奶", "葡萄", "樱桃", "面包", "黄瓜", "香蕉", "音乐",
    "我", "特别", "喜欢", "吃", "苹果", "和", "水果", "你", "吃饭", "梨子",
    "喝", "水", "果汁", "西瓜", "橙子", "超市", "购物", "蔬菜", "胡萝卜", "生菜",
    "做", "饼干", "巧克力", "糖果", "冰淇淋", "果冻", "健康", "营养", "美味", "早餐",
    "午餐", "晚餐", "零食", "烹饪", "食谱", "学习", "工作", "休息", "运动", "旅行",
    "电影", "书籍", "游戏", "快乐", "悲伤", "愤怒", "惊讶", "无聊", "好奇", "兴奋",
    "紧张", "天气", "晴朗", "下雨", "雪", "风", "热", "冷", "温暖", "凉爽",
    "城市", "乡村", "山", "海", "河流", "公园", "花园", "动物园", "学校", "医院",
    "银行", "商店", "市场", "机场", "车站", "图书馆", "博物馆", "汽车", "自行车",
    "火车", "船", "公交车", "出租车", "步行", "计算机", "手机", "电视", "相机",
    "耳机", "手表", "冰箱", "微波炉", "著名", "景点", "桥梁"
}

# 姓氏库（简化版，可扩展）
surnames = set(["张", "李", "王", "赵", "刘", "陈", "杨", "黄", "吴", "周", "徐", "孙", "江"])

# 示例语料统计数据（可通过大语料训练替换）
# 假设从语料中统计得到的字频、姓名频率数据
name_char_freq = {
    "建": {"freq": 100, "first_prob": 0.8, "last_prob": 0.2},
    "国": {"freq": 120, "first_prob": 0.4, "last_prob": 0.8},
    "庆": {"freq": 80, "first_prob": 0.2, "last_prob": 0.8},
    "江": {"freq": 60, "first_prob": 0.5, "last_prob": 0.5},
    "高": {"freq": 23, "first_prob": 0.5, "last_prob": 0.5},
    "林": {"freq": 15, "first_prob": 0.3, "last_prob": 0.7},
    "大": {"freq": 15, "first_prob": 0.3, "last_prob": 0.7},
    "桥": {"freq": 15, "first_prob": 0.3, "last_prob": 0.7},
}

# 阈值表
surname_threshold = {"张": -3.0, "李": -3.2, "王": -3.1, "江": -3.1}

# 逆向最大匹配（BMM）分词
def backward_maximum_matching(text, word_dict):
    result = []
    max_dict_length = max(len(word) for word in word_dict)
    while text:
        for i in range(min(len(text), max_dict_length), 0, -1):
            word = text[-i:]
            if word in word_dict:
                result.insert(0, word)
                text = text[:-i]
                break
        else:
            result.insert(0, text[-1])
            text = text[:-1]
    return result

# 人名概率估计公式
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

# 人名识别函数
def recognize_names(text, word_dict, threshold=0.0001):
    segmented = backward_maximum_matching(text, word_dict)
    potential_names = []
    for i, word in enumerate(segmented):
        if word in surnames:  # 姓氏触发点
            # 寻找潜在名字
            if i + 1 < len(segmented):  # 至少需要两个字
                m1 = segmented[i + 1]
                m2 = segmented[i + 2] if i + 2 < len(segmented) and len(segmented[i + 2]) == 1 else "" # 可能有第三个字
                if len(m1) + len(m2) >= 1 and len(m1) + len(m2) <=2:  # 有界范围
                    prob = calculate_name_probability(word, m1, m2)
                    if prob > surname_threshold[word]:  # 判断是否满足概率条件
                        potential_names.append((word + m1 + m2, prob))
    # 去重和规则修正
    final_names = []
    for name, prob in potential_names:
        if all([
            not name.isdigit(),  # 修饰规则：避免数字
            name not in word_dict,  # 避免已有词典中的词
        ]):
            final_names.append(name)
    return segmented, final_names

# 测试数据
text_list = [
        "张建国正在演讲。",
        "李国庆发表了意见。",
        "南京市长江大桥是著名景点。"
    ]
# 测试
print("====================================人名识别测试====================================")
for text in text_list:
    print(f"输入文本: {text}")
    segmented, names = recognize_names(text, word_dict)
    print(f"分词结果: {segmented}")
    print(f"识别的人名: {names}")
    print()
