# health_recommendation.py
# 个性化健康建议：基于用户数据的健康分析
# 个性化健康建议：利用线性回归模型生成健康建议。
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# 模拟用户数据（步数、睡眠时长、卡路里摄入）
data = pd.DataFrame({
    'steps': np.random.randint(3000, 15000, size=100),
    'sleep_hours': np.random.uniform(4, 10, size=100),
    'calories': np.random.uniform(1500, 3000, size=100),
    'health_score': np.random.uniform(50, 100, size=100)
})

# 特征和目标
X = data[['steps', 'sleep_hours', 'calories']]
y = data['health_score']

# 模型训练
model = LinearRegression()
model.fit(X, y)

# 个性化建议
def generate_recommendation(steps, sleep_hours, calories):
    predicted_score = model.predict([[steps, sleep_hours, calories]])[0]
    if predicted_score > 85:
        return "健康状况良好，继续保持！"
    elif 70 <= predicted_score <= 85:
        return "注意适量运动和均衡饮食，以提升健康分数。"
    else:
        return "健康状况较差，请增加锻炼并调整饮食习惯。"

# 示例
print(generate_recommendation(8000, 7, 2000))
