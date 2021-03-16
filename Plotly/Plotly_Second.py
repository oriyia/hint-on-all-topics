import pandas as pd
import plotly.express as px
import plotly.offline as offline

df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})
print(df)

color = {'backgroud': '#111111',
         'text': '#7FDBFF'}

fig = px.bar(df, x='Fruit', y='Amount', barmode='group', color='City')
fig.update_layout(
    plot_
)
# offline.plot(fig)