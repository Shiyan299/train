# import packages
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# 任务1：EDA分析（在Streamlit中展示）
# ----------------------------
def perform_eda(df):
    st.subheader('📊 探索性数据分析 (EDA)')
    
    # 1. 数据基本信息
    st.write('### 1. 数据基本结构')
    col1, col2 = st.columns(2)
    with col1:
        st.write('前5行数据：')
        st.dataframe(df.head())
    with col2:
        st.write('数据维度：', df.shape)
        st.write('列信息：')
        st.dataframe(df.dtypes.reset_index().rename(columns={'index': '列名', 0: '数据类型'}))
    
    # 2. 缺失值分析
    st.write('### 2. 缺失值统计')
    missing = df.isnull().sum().reset_index()
    missing.columns = ['列名', '缺失值数量']
    missing['缺失比例(%)'] = (missing['缺失值数量'] / len(df)) * 100
    st.dataframe(missing.style.format({'缺失比例(%)': '{:.2f}%'}))
    
    # 3. 关键变量统计描述
    st.write('### 3. 数值变量统计描述')
    st.dataframe(df[['Age', 'Fare', 'SibSp', 'Parch']].describe().round(2))
    
    # 4. 存活率与分类变量关系
    st.write('### 4. 存活率与关键变量关系')
    # 4.1 按客舱等级（Pclass）
    survival_pclass = df.groupby('Pclass')['Survived'].mean() * 100
    # 4.2 按性别（Sex）
    survival_sex = df.groupby('Sex')['Survived'].mean() * 100
    # 4.3 按是否有兄弟姐妹/配偶（SibSp）
    survival_sibsp = df.groupby('SibSp')['Survived'].mean() * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.write('按客舱等级：')
        st.dataframe(survival_pclass.reset_index().rename(
            columns={'Pclass': '客舱等级', 'Survived': '存活率(%)'}
        ).style.format({'存活率(%)': '{:.1f}%'}))
    with col2:
        st.write('按性别：')
        st.dataframe(survival_sex.reset_index().rename(
            columns={'Sex': '性别', 'Survived': '存活率(%)'}
        ).style.format({'存活率(%)': '{:.1f}%'}))


# ----------------------------
# 任务2：Streamlit数据应用
# ----------------------------
def main():
    # 设置页面标题（替换为你的全名）
    st.title('🚢 Titanic Data Application-Zhang Shiyan')
    st.write('基于Titanic数据集的生存分析与可视化工具')
    
    # 1. 读取数据
    try:
        # 确保train.csv与代码在同一文件夹
        df = pd.read_csv('train.csv')
        st.success('数据读取成功！共包含 {} 条记录'.format(len(df)))
    except FileNotFoundError:
        st.error('未找到train.csv文件，请确保数据文件与代码在同一目录')
        return
    
    # 2. 数据预处理（处理缺失值）
    df['Age'].fillna(df['Age'].median(), inplace=True)  # 年龄用中位数填充
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # 登船港口用众数填充
    
    # 3. 侧边栏筛选器
    st.sidebar.header('🔍 数据筛选')
    # 3.1 客舱等级
    pclass = st.sidebar.multiselect(
        '选择客舱等级',
        options=df['Pclass'].unique(),
        format_func=lambda x: f'第{x}等舱',
        default=df['Pclass'].unique()
    )
    # 3.2 性别
    sex = st.sidebar.radio(
        '选择性别',
        options=df['Sex'].unique(),
        default='male'
    )
    # 3.3 年龄范围
    age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
    age_range = st.sidebar.slider(
        '选择年龄范围',
        min_value=age_min,
        max_value=age_max,
        value=(age_min, age_max)
    )
    
    # 4. 应用筛选条件
    filtered_df = df[
        (df['Pclass'].isin(pclass)) &
        (df['Sex'] == sex) &
        (df['Age'] >= age_range[0]) &
        (df['Age'] <= age_range[1])
    ]
    
    # 5. 展示筛选后的数据
    st.subheader('🔬 筛选后的数据')
    st.dataframe(filtered_df, height=300)
    
    # 6. 可视化
    st.subheader('📈 数据可视化')
    # 6.1 存活率按客舱等级（柱状图）
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Pclass', y='Survived', data=df, ax=ax1, palette='viridis')
    ax1.set_title('各客舱等级的存活率', fontsize=12)
    ax1.set_xlabel('客舱等级')
    ax1.set_ylabel('存活率')
    ax1.set_xticklabels(['1等舱', '2等舱', '3等舱'])
    st.pyplot(fig1)
    
    # 6.2 年龄分布（直方图）
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.histplot(filtered_df['Age'], bins=15, kde=True, ax=ax2, color='skyblue')
    ax2.set_title('筛选后乘客的年龄分布', fontsize=12)
    ax2.set_xlabel('年龄')
    ax2.set_ylabel('人数')
    st.pyplot(fig2)
    
    # 6.3 存活率与票价关系（散点图）
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x='Fare', y='Age', hue='Survived', data=filtered_df, ax=ax3, 
                   palette={0: 'red', 1: 'green'})
    ax3.set_title('票价与年龄的关系（颜色表示存活状态）', fontsize=12)
    ax3.set_xlabel('票价（美元）')
    ax3.set_ylabel('年龄')
    ax3.legend(title='存活状态', labels=['未存活', '存活'])
    st.pyplot(fig3)
    
    # 7. 展示EDA结果（任务1）
    perform_eda(df)


if __name__ == '__main__':
    main()