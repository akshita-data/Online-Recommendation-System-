import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify



df = pd.read_excel('OnlineRetail.xlsx') 


df['StockCode'] = df['StockCode'].astype(str)

#  Create User-Item Matrix
user_item = df.pivot_table(
    index='CustomerID',
    columns='StockCode',
    values='Quantity',
    aggfunc='sum',
    fill_value=0
)

# Transpose to Item-User Matrix
item_user = user_item.T

#  Compute Cosine Similarity between items
item_sim = pd.DataFrame(
    cosine_similarity(item_user),
    index=item_user.index,
    columns=item_user.index
)

#  Recommendation Function
def recommend_products(stock_code, n=5):
    stock_code = str(stock_code)
    
    if stock_code not in item_sim.index:
        return f" Error: StockCode '{stock_code}' not found in similarity matrix."
    
    sims = item_sim.loc[stock_code].drop(stock_code)
    top = sims.nlargest(n)
    
    recs = (
        pd.DataFrame({
            'StockCode': top.index,
            'Similarity': top.values
        })
        .merge(
            df[['StockCode', 'Description']].drop_duplicates(),
            on='StockCode',
            how='left'
        )
    )
    return recs

#  Example Usage
print(" Available StockCodes:", item_sim.index[:10].tolist())  # Preview some codes
print(recommend_products('85099B', n=5))  # Replace with a valid StockCode from your data

#  Flask App
app = Flask(__name__)

@app.route('/recommend', methods=['GET'])
def recommend_endpoint():
    code = request.args.get('code')
    n = int(request.args.get('n', 5))
    results = recommend_products(code, n)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
