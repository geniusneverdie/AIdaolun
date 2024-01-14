from kanren import *
from kanren.core import lall     # lall 包用于定义规则
import time
#定义 left()函数，用来查找哪个房屋左边
def left(q, p, list):
     return membero((q,p), zip(list, list[1:]))
#定义 next()函数，用来接近谁的房子
def next(q, p, list):
    return conde([left(q, p, list)], [left(p, q, list)])
from kanren import *
from kanren.core import lall     # lall 包用于定义规则
import time

#定义 left()函数，用来查找哪个房屋左边
def left(q, p, list):
     return membero((q,p), zip(list, list[1:]))
#定义 next()函数，用来接近谁的房子
def next(q, p, list):
    return conde([left(q, p, list)], [left(p, q, list)])

houses = var()
rules_zebraproblem = lall(
     (eq, (var(), var(), var(), var(), var()), houses),       # 5 个 var()分别代表人、烟、饮料、动物、屋子颜色
      # 房子里的每个子成员有五个属性: membero(国家，身份，饮料，宠物，房子)
     (membero,('英国人', var(), var(), var(), '红色'), houses),          # 1. 英国人住在红色的房子里
     (membero,('西班牙人', var(), var(), '狗', var()), houses),          # 2. 西班牙人养了一条狗
         (membero,('日本人', '油漆工', var(), var(), var()), houses),        # 3. 日本人是一个油漆工
         (membero,('意大利人', var(), '茶', var(), var()), houses),          # 4. 意大利人喜欢喝茶
        # 5. 挪威人住在左边的第一个房子里
         (eq,(('挪威人', var(), var(), var(), var()), var(), var(), var(), var()), houses),
         (left,(var(), var(), var(), var(), '白色'),(var(), var(), var(), var(), '绿色'), houses),    # 6. 绿房子在白房子的右边
         (membero,(var(), '摄影师', var(), '蜗牛', var()), houses),                     # 7. 摄影师养了一只蜗牛
         (membero,(var(), '外交官', var(), var(), '黄色'), houses),                     # 8. 外交官住在黄房子里
         (eq,(var(), var(), (var(), var(), '牛奶', var(), var()), var(), var()), houses),      # 9. 中间那个房子的人喜欢喝牛奶
         (membero,(var(), var(), '咖啡', var(), '绿色'), houses),                  # 10. 喜欢喝咖啡的人住在绿房子里
        # 11. 挪威人住在蓝色的房子旁边
         (next,('挪威人', var(), var(), var(), var()),(var(), var(), var(), var(), '蓝色'), houses),
         (membero,(var(), '小提琴家', '橘子汁', var(), var()), houses),               # 12. 小提琴家喜欢喝橘子汁
        # 13. 养狐狸的人所住的房子与医生的房子相邻
        (next,(var(), var(), var(), '狐狸', var()),(var(), '医生', var(), var(), var()), houses),
        # 14. 养马的人所住的房子与外交官的房子相邻
         (next,(var(), var(), var(), '马', var()),(var(), '外交官', var(), var(), var()), houses),
        (membero,(var(), var(), var(), '斑马', var()), houses),                  # 问题 1. 有人养斑马
         (membero,(var(), var(), '矿泉水', var(), var()), houses),                   # 问题 2. 有人喜欢喝矿泉水
    )

# 使用 rules_zebraproblem 约束运行解算器
solutions = run(0, houses, rules_zebraproblem)
# 提取解算器的输出
output = [house for house in solutions[0] if '斑马' in house][0][4]
print ('\n{}房子里的人养斑马'.format(output))
output = [house for house in solutions[0] if '矿泉水' in house][0][4]
print ('\n{}房子里的人喜欢喝矿泉水\n'.format(output))

# 解算器的输出结果展示
print("所有结果如下:")
for i in solutions[0]:
    print(i)
