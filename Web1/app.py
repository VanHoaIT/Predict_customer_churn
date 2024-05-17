from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Nạp mô hình đã lưu
model_file_path = './model2.pkl'
with open(model_file_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu từ form
    Total_Trans_Ct = int(request.form['Total_Trans_Ct'])
    Total_Trans_Amt = int(request.form['Total_Trans_Amt'])
    Total_Revolving_Bal = int(request.form['Total_Revolving_Bal'])
    Total_Ct_Chng_Q4_Q1 = int(request.form['Total_Ct_Chng_Q4_Q1'])
    Avg_Utilization_Ratio = float(request.form['Avg_Utilization_Ratio'])
    Total_Amt_Chng_Q4_Q1 = int(request.form['Total_Amt_Chng_Q4_Q1'])
    Total_Relationship_Count = int(request.form['Total_Relationship_Count'])

    # Tạo mảng numpy cho dữ liệu đầu vào
    input_data = np.array([[Total_Trans_Ct, Total_Trans_Amt, Total_Revolving_Bal,
                            Total_Ct_Chng_Q4_Q1, Avg_Utilization_Ratio, 
                            Total_Amt_Chng_Q4_Q1, Total_Relationship_Count]])

    # tính phần trăm khả năng xảy ra
    probability = model.predict_proba(input_data)[:,1]*100
    probability[0] = round(probability[0],2)

    # Thực hiện dự đoán
    prediction = model.predict(input_data)[0]

    # Trả kết quả dự đoán
    if prediction == 1:
        result = "Khách hàng sẽ rời đi."
    else:
        result = "Khách hàng sẽ ở lại."

    return render_template('result.html', prediction_text=result,
                           probability_text = probability[0],
                           Total_Trans_Ct=Total_Trans_Ct,
                           Total_Trans_Amt=Total_Trans_Amt,
                           Total_Revolving_Bal=Total_Revolving_Bal,
                           Total_Ct_Chng_Q4_Q1=Total_Ct_Chng_Q4_Q1,
                           Avg_Utilization_Ratio=Avg_Utilization_Ratio,
                           Total_Amt_Chng_Q4_Q1=Total_Amt_Chng_Q4_Q1,
                           Total_Relationship_Count=Total_Relationship_Count)

if __name__ == "__main__":
    app.run(debug=True)
