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
from states import state_dict
from observations import *
import streamlit.components.v1 as components
from streamlit_extras.stylable_container import stylable_container
#143c94,#1c9c94,#cfe6da

# sudo apt-get install graphviz
# sudo apt install font-manager
st.set_page_config(layout="wide")
css_styles="""
                {
                    border: 1px solid #cfe6da;
                    border-radius: 1rem;
                    padding:calc(1em - 1px);
                    padding-left:calc(1em - 2px);
                    background-color:#cfe6da;
                    
                }
                """

btn_labels = ['Questions 1 and 2','Question 4']

if "btn_prsd_status" not in st.session_state:
    st.session_state.btn_prsd_status = [True]+[False] * (len(btn_labels)-1)

        
unpressed_colour = "#cfe6da"
pressed_colour = "#143c94"

def ChangeButtonColour(widget_label, prsd_status):
    btn_bg_colour = pressed_colour if prsd_status == True else unpressed_colour
    text_color = 'white' if prsd_status == True else '#143c94'
    htmlstr = f"""
        <script>
            var elements = window.parent.document.querySelectorAll('button');
            for (var i = 0; i < elements.length; ++i) {{ 
                if (elements[i].innerText == '{widget_label}') {{ 
                    elements[i].style.background = '{btn_bg_colour}';
                    elements[i].style.color = '{text_color}';
                }}
            }}
        </script>
        """
    components.html(f"{htmlstr}", height=0, width=0)

def ChkBtnStatusAndAssignColour():
    for i in range(len(btn_labels)):
        ChangeButtonColour(btn_labels[i], st.session_state.btn_prsd_status[i])

def btn_pressed_callback(x):
    i=btn_labels.index(x)+1
    st.session_state.btn_prsd_status = [False] * len(btn_labels)
    st.session_state.btn_prsd_status[i-1] = True





if 'page' not in st.session_state:
    st.session_state['page'] = 'Questions 1 and 2'

def q(x):
   return state_dict[st.session_state['page']][x]  
# st.write(q('x_cols')) 
def changeState(x):
    btn_pressed_callback(x)
    st.session_state['page']=x


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
    
    with st.container():
        st.markdown("""<style>
            div[data-testid='stVerticalBlock']:has(div#chat_inner):not(:has(div#chat_outer)) {background-color: #E4F2EC};
            </style>
            """, unsafe_allow_html=True)
        statess=['Questions 1 and 2','Question 4']
        for m in statess:
             st.button(m,type="primary",use_container_width=True,on_click=changeState,args=(m,))  #lambda:changeState(m)
        ChkBtnStatusAndAssignColour()   
          

var=st.selectbox('Main column to analyze:',tuple([i for i in df.columns[1:] if i!='y']),[i for i in df.columns[1:] if i!='y'].index(q('var')))
x_cols=st.multiselect("Columns to work with:",list([i for i in df.columns[1:] if i!='y']),q('x_cols'),)  #['Institucion','Año','Curso','Carrera 1 Clustered','Gender']
# var=x_cols[0] if len(x_cols)>0 else 'Carrera 1 Clustered'
df['Number of Cases']=1.
color='University'
bar1=px.bar(df.groupby([var,color],as_index=False).sum().sort_values('Number of Cases',ascending=False),x=var,y='Number of Cases',color=color,color_discrete_sequence=color_palette)
bar2=px.bar(df.groupby([var],as_index=False).mean(numeric_only=True).sort_values('y',ascending=False),x=var,y='y',color_discrete_sequence=color_palette)
scatter=px.scatter(df.groupby([var],as_index=False).agg({'y':'mean','Number of Cases':'sum'}),x='Number of Cases',y='y',color=var,size='Number of Cases',log_x=True,color_discrete_sequence=color_palette)
st.write('---')
if obs_dict1[st.session_state['page']]!='':
    with stylable_container(key="container_with_border",css_styles=css_styles,):      
            st.markdown('<p style="width: 92%;">'+obs_dict1[st.session_state['page']]+'</p>',unsafe_allow_html=True)
www='70' if st.session_state['page']=='Question 4' else '92'

if obs_dict1[st.session_state['page']]!='':
    with stylable_container(key="container_with_border",css_styles=css_styles,):      
            st.markdown(f'<p style="width: {www}%;">'+obs_dict2[st.session_state['page']]+'</p>',unsafe_allow_html=True)



col1, col2 = st.columns(2)
col1.subheader(f'Chosen University By "{var}"')
col1.plotly_chart(bar1, use_container_width=True)
col2.subheader(f'Chosen University Share By "{var}"')
col2.plotly_chart(bar2, use_container_width=True)
st.write('---')
st.subheader(f'Size of "{var}" against University Share')
st.plotly_chart(scatter, use_container_width=True)
st.write('---')
st.subheader('Path Of Chosen Variables')
san=sankey(df,x_cols+['University'],'Number of Cases',500,10000,-1)
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
st.write('---')
st.subheader('Tree Path Understanding')
left_co2, cent_co2,last_co2,other2 ,ja2,last2= st.columns(6)
with cent_co2:
    st.image('mini_pred.svg',width=600) #use_column_width=True
# st.pyplot(fig)


# x_cols=['Gender','Año','Institucion','Curso','Carrera 1 Clustered']
exclude=[]#['Gender_Male','Institucion_Insurgentes León','Curso_BIV EN INF ADM','Carrera 1 Clustered_Law']
meta_df=pd.get_dummies(df[x_cols], dtype='float')
no_corr=[]
for j in x_cols:
  no_corr=no_corr+ [i for i in meta_df.columns if (j in i) and (i not in exclude)]
X=meta_df[no_corr].to_numpy()
X = sm.add_constant(X)
names=['constant']+no_corr #
y=df['y'].to_numpy()

errorio=True
alpha=0.01
while errorio:
    try:
        alpha=alpha+0.1
        results = sm.Logit(y, X).fit_regularized(alpha=alpha)  #fit
        # results = sm.MNLogit(y, X).fit_regularized(alpha=alpha+0.1)
        errorio=False
    except:
        pass
# print(alpha+0.5)
st.write('---')
st.subheader('Z-values Of Variables')
st.write(results.summary(xname=names))


# from sklearn.linear_model import LogisticRegression

# clf = LogisticRegression(random_state=0).fit(X, y)
# st.write({names[index]:i for index,i in enumerate(clf.coef_[0] )})