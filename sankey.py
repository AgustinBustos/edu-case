import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from color_palette import color_palette
# l=[px.colors.qualitative.Alphabet for i in range(100)]
# colores=px.colors.qualitative.Plotly+px.colors.qualitative.D3+px.colors.qualitative.G10+px.colors.qualitative.T10+px.colors.qualitative.Alphabet+[item for sublist in l for item in sublist]

colores=100*color_palette
# colores.reverse()
def sankey(df_pandas,categorical_dimensions,ys,height,width,colorOn):
  first_group=df_pandas.groupby(categorical_dimensions).sum().reset_index()
  rowQuantity=1000
  first_group["q"]=np.floor(rowQuantity*(first_group[ys]/first_group[ys].sum()))
  first_group["color"]=0
  counter=0
  for i in first_group[categorical_dimensions[colorOn]]:
    #first_group.loc[first_group[categorical_dimensions[-1]]==i,"color"]=counter/len(first_group[categorical_dimensions[-1]].unique())
    first_group.loc[first_group[categorical_dimensions[colorOn]]==i,"color"]=colores[counter]
    counter=counter+1

  first_group=first_group.loc[first_group.index.repeat(first_group.q)]
  dimensions = [dict(values=first_group[label], label=label) for label in categorical_dimensions]
  # Build colorscale
  #color = np.zeros(len(first_group), dtype='uint8')

  color = first_group["color"].tolist()

  colorscale = [[0, 'gray'], [1, 'firebrick']]

  # Build figure as FigureWidget
  fig = go.FigureWidget(
      data=[ go.Parcats(
          # domain={'y': [0, 0.4]},
          dimensions=dimensions,
          line={
              # 'colorscale': colorscale,
              # 'cmin': 0,
              #   'cmax': 1,
                'color': color,    #acaaaaaa
                'shape': 'hspline'})
      ])

  fig.update_layout(

          height=height,
          width=width,
          margin={'t':20,'l':150,'b':20,'r':0},
          # xaxis={'title': 'Horsepower'},
          # yaxis={'title': 'MPG', 'domain': [0.6,1]},
          dragmode='lasso', hovermode='closest',
          font=dict(
                family="Sans Serif",
                size=13,  # Set the font size here
                color="#4B2E2A"
            ))



  return fig
# sankey(df,x_cols+['University'],'ones',500,1000,-1)