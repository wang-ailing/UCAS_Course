# 有限词表
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
    "耳机", "手表", "冰箱", "微波炉"
}

def backward_maximum_matching(text, word_dict):
    result = []
    max_dict_length = max(len(word) for word in word_dict)
    max_length = min(len(text), max_dict_length)
    while text:
        # 遍历长度，从最长的词开始匹配
        for i in range(max_length, 0, -1):
            word = text[-i:]
            if word in word_dict:
                result.insert(0, word)
                text = text[:-i]
                break
        else: # 遍历完所有长度，没有匹配到词，则把最后一个字母加到结果中
            result.insert(0, text[-1])
            text = text[:-1]
    
    return result

def evaluate(predicted, target):
    predicted_set = set(predicted)
    target_set = set(target)
    
    true_positive = len(predicted_set & target_set)
    precision = true_positive / len(predicted_set) if predicted_set else 0 # 预测正确的词数 / 预测词数
    recall = true_positive / len(target_set) if target_set else 0 # 预测正确的词数 / 目标词数
    f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return precision, recall, f_measure

# 测试示例 1 
text = "我喜欢吃香蕉"
# 使用逆向最大分词算法进行分词
predicted = backward_maximum_matching(text, word_dict)
# 目标分词结果
target = ["我", "喜欢", "吃", "香蕉"]



# 输出分词结果
print("程序分词结果:", predicted)

# 计算准确率、召回率和F-测度
precision, recall, f_measure = evaluate(predicted, target)
print(f"准确率: {precision:.2f}, 召回率: {recall:.2f}, F-测度: {f_measure:.2f}")


# 测试示例 2
text = "我喜欢吃饭"
# 使用逆向最大分词算法进行分词
predicted = backward_maximum_matching(text, word_dict)
# 目标分词结果
target = ["我", "喜欢", "吃饭"]



# 输出分词结果
print("程序分词结果:", predicted)

# 计算准确率、召回率和F-测度
precision, recall, f_measure = evaluate(predicted, target)
print(f"准确率: {precision:.2f}, 召回率: {recall:.2f}, F-测度: {f_measure:.2f}")

# 测试示例 3
text = "王五喜欢吃饭"
# 使用逆向最大分词算法进行分词
predicted = backward_maximum_matching(text, word_dict)
# 目标分词结果
target = ["王五", "喜欢", "吃饭"]

# 输出分词结果
print("程序分词结果:", predicted)

# 计算准确率、召回率和F-测度
precision, recall, f_measure = evaluate(predicted, target)
print(f"准确率: {precision:.2f}, 召回率: {recall:.2f}, F-测度: {f_measure:.2f}")