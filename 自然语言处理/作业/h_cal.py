import math

text = """
遥远的夜空，有一个弯弯的月亮

弯弯的月亮下面，是那弯弯的小桥

小桥的旁边，有一条弯弯的小船

弯弯的小船悠悠，是那童年的阿娇

阿娇摇着船，唱着那古老的歌谣

歌声随风飘，飘到我的脸上

脸上淌着泪，像那条弯弯的河水

弯弯的河水流啊，流进我的心上"""


text = text.replace("\n", "")
text = text.replace("，", "")
char_list = list(text)
total_char = len(char_list)
print(len(char_list))
count_dict = {}
for char in char_list:
    if char in count_dict:
        count_dict[char] += 1
    else:
        count_dict[char] = 1

print(len(count_dict))

total_h_value = 0
for key, value in count_dict.items():
    p_value = value / total_char
    h_value = -1 * (p_value * math.log(p_value, 2))
    total_h_value += h_value

print(total_h_value)

# char_list = set(char_list)
# print(len(char_list))
# print(char_list)