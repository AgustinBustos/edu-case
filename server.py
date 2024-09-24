import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import dtreeviz
import numpy as np
import statsmodels.api as sm
import streamlit as st
from sankey import sankey
import numpy as np
from color_palette import color_palette
#143c94,#1c9c94

# sudo apt-get install graphviz
# sudo apt install font-manager
st.set_page_config(layout="wide")

st.title('Edu-Case')
df=pd.read_csv('Informe Alumnos Insurgentes Processed.csv')
df=df.fillna('None')
# df.drop(['Email', 'Nombre y Apellido'],axis=1).to_csv('Informe Alumnos Insurgentes Processed2.csv')

# var='Carrera 1 Clustered'

with st.sidebar:
    left_co, cent_co,last_co,other ,ja,last= st.columns(6)
    with cent_co:
        st.image('uin.svg',width=170)
    st.write('')
    with st.expander("Visualization:"):
        optionss=('Manufacturer', 'Brand', 'Channel', 'Sub Channel', 'Variable')
        optionss2=(  'Investment (R$)',
            'Comunicação (One Score)', 
            'Número de eventos ',
            'Número de Participantes em Eventos',  
            'GRP',
            'Impressions',
            'Faces/Telas',
            'Number of Installations')

        # by = st.selectbox(
        #     "By:",
        #     optionss,index=optionss.index(q('by')))
        # colors = st.selectbox(
        #     "Color:",
        #     optionss,optionss.index(q('colors')))
        # selected_metric = st.selectbox(
        #     "Metric:",optionss2,optionss2.index(q('selected_metric'))
        #     )  
    with st.container():
        st.markdown("""<style>
            div[data-testid='stVerticalBlock']:has(div#chat_inner):not(:has(div#chat_outer)) {background-color: #E4F2EC};
            </style>
            """, unsafe_allow_html=True)
        statess=['Market Total Marketing Investment','Total Investment by Channel','Total Investment by Brand','Ambev Total Investment','Ambev Trade Investment Desagregation','Ambev Events Investment Desagregation','Ambev Media Investment Desagregation','Ambev Trade Materials','Ambev Events Number','Ambev Media GRP','Ambev Media Impressions','Ambev Media Faces/Telas']
        # for m in statess:
        #      st.button(m,type="primary",use_container_width=True,on_click=changeState,args=(m,))  #lambda:changeState(m)
        # ChkBtnStatusAndAssignColour()   
          

var=st.selectbox('Main column to analyze',tuple([i for i in df.columns[1:] if i!='y']))
x_cols=st.multiselect("Columns to work with:",list([i for i in df.columns[1:] if i!='y']),['Institucion','Año','Curso','Carrera 1 Clustered','Gender'],)
# var=x_cols[0] if len(x_cols)>0 else 'Carrera 1 Clustered'
df['ones']=1.
color='University'
bar1=px.bar(df.groupby([var,color],as_index=False).sum().sort_values('ones',ascending=False),x=var,y='ones',color=color,color_discrete_sequence=color_palette)
bar2=px.bar(df.groupby([var],as_index=False).mean(numeric_only=True).sort_values('y',ascending=False),x=var,y='y',color_discrete_sequence=color_palette)
scatter=px.scatter(df.groupby([var],as_index=False).agg({'y':'mean','ones':'sum'}),x='ones',y='y',color=var,size='ones',log_x=True,color_discrete_sequence=color_palette)

col1, col2 = st.columns(2)
col1.plotly_chart(bar1, use_container_width=True)
col2.plotly_chart(bar2, use_container_width=True)
st.plotly_chart(scatter, use_container_width=True)

san=sankey(df,x_cols+['University'],'ones',500,10000,-1)
st.plotly_chart(san, use_container_width=True)





meta_df=pd.get_dummies(df[x_cols], dtype='float')
x_cols_2=list(meta_df.columns)
meta_df['y']=df['y'].astype(int)
# %%capture
# iris = load_iris()

X_meta = meta_df[x_cols_2].to_numpy()
y_meta = meta_df['y'].to_numpy()

# clf = DecisionTreeClassifier(max_depth=5)
#first 1e-3
nodes=[]
space=np.logspace(0,-5,num=1000)
for cut in space:
  clf = DecisionTreeClassifier(min_impurity_decrease=cut,min_samples_leaf=10)
  clf.fit(X_meta, y_meta)
  nodes.append(clf.tree_.node_count)


try:
  cutter=space[1:][np.argmax(np.diff(nodes))-1]#+1e-10
#   cutter=1e-10
except:
  cutter=1e-10
# print('-----------------------------')
# print(cutter)
clf = DecisionTreeClassifier(min_impurity_decrease=cutter,min_samples_leaf=10)
clf.fit(X_meta, y_meta)
# nodes.append(clf.tree_.node_count)
viz_model = dtreeviz.model(clf,
                           X_train=X_meta, y_train=y_meta,
                           feature_names=x_cols_2,
                           target_name='University',
                           class_names=['other', 'own',])

v = viz_model.view()
# viz_model.figure
# fig = viz_model.get_figure()
# fig.patch.set_facecolor('#fafafa')
# v.show()
# displayHTML(v.svg())
v.save("mini_pred.svg")
file_path="mini_pred.svg"
with open(file_path, 'r', encoding='utf-8') as file:
    svg_content = file.read()

# Replace the old string with the new string
updated_content = svg_content.replace('fill: #ffffff', 'fill: #fafafa').replace('fill="white"', 'fill="#fafafa"')

# Write the updated content back to the SVG file
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(updated_content)
# st.markdown(
#     """
#     <style>
#         button[title^=Exit]+div [data-testid=stImage]{
#             text-align: center;
#             display: block;
#             margin-left: auto;
#             margin-right: auto;
#             width: 100%;
#         }
#     </style>
#     """, unsafe_allow_html=True
# )
st.image('mini_pred.svg',width=600) #use_column_width=True
# st.pyplot(fig)


# x_cols=['Gender','Año','Institucion','Curso','Carrera 1 Clustered']
exclude=['Gender_Male','Institucion_Insurgentes León','Curso_BIV EN INF ADM','Carrera 1 Clustered_Law']
meta_df=pd.get_dummies(df[x_cols], dtype='float')
no_corr=[]
for j in x_cols:
  no_corr=no_corr+ [i for i in meta_df.columns if (j in i) and (i not in exclude)]
X=meta_df[no_corr].to_numpy()

y=df['y'].to_numpy()

results = sm.Logit(y, X).fit()
st.write(results.summary(xname=no_corr))