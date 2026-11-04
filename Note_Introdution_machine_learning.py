#This is cheat sheet of my process in learning Pandas & Machine Learning\

import pandas as pd #Khai báo thư viện pandas
from sklearn.tree import DecisionTreeRegressor
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv' # Đường dẫn file excel
melbourne_data = pd.read_csv(melbourne_file_path) # pd.read_cvs (read file excel)
melbourne_data.describe() # Thống kê dữ liệu trong đó:
#Count: Số lượng giá trị
#Mean: Giá trị trung bình
#std: Độ lệch chuẩn cho biết dữ liệu phân tán nhiều hay ít
#min: Gtri nhỏ nhất
#max: Gtri lớn nhất
#25%,50%,75%: Tứ phân vị

melbourne_data.columns # in ra các cột đầu tiên của data frame
# Machine Learning
# Selection Prediciton Target & choosing features

# Selection Prediction Target

# Nếu muốn dự đoán giá nhà thì
y = melbourne_data['Price'] # <-- Selection Prediction Target Lấy dữ liệu của cột giá nhà

# Choosing Features <- Chọn Đặc trưng đầu vào
# Thông tin mô hình lựa chọn để dựa đoán
# Chọn số phòng khoảng cách, đến trung tâm và diện tích đất để dự đoán nhà:
features = ['Rooms', 'Distance', 'Landsize']
X = melbourne_data[features]

# Lưu ý sự khác nhau giữa cách lấy cột
# df['Price']                # Một cột, thường trả về Series
# df[['Rooms', 'Distance']]  # Nhiều cột, trả về DataFrame

# Review
# Trước khi building một model nên review lại
print(X.describe()) #<- Thống kê
print(X.head()) # <- In 5 dòng đầu tiên
print(X) # <- In toàn độ dòng (Data Frame)
# Matching Selection prediction Target & Choosing Features
print(X.head())
print(y.head())
# Trong đó
# X chứa Room, distance, landsize <- Vai trò: đầu vào
# Y Price: Đầu ra cần dự đoán

# Khi có data rồi thì sẽ đưa x và y để huấn luyện mô hình, trong thư viện scikit-learn:
melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(X, y)

# Sau khi huấn luyện mô hình dữu liệu rồi sẽ
# Make Predictions
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))

# Sau khi huấn luyện & đưa ra dự đoán rồi sẽ
# Model Validation Kiểm chứng mô hình
# Note: Xây dựng được mô hình dự đoán rồi nhưng chưa chắc mô hình đó có chất lượng hay không ?
# Vc Kiểm chứng mô hình là tiền đề để cải thiện model trong các vòng lặp lần thử khác
# Có nhiều cách đánh giá chất lượng mô hình nhưng sẽ bắt đầu với cách basic nhất
#Gọi là Sai số tuyệt đối trung bình (Meaning Absolute Error) = (MAE)
#Nếu một  căn nhà có giá là 150 000$ thì bạn dự đoán là 100 000 => MAE = 50 000 -> 150 000 - 100 000
# MAE => Thước đo chất lượng mô hình
# Cách CACULATE MAE
# from sklearn.metrics import mean_absolute_error
    #predicted_home_prices = melbourne_model.predict(X)
    #mean_absolute_error(y, predicted_home_prices)
# Gặp mô hình dữ liệu cũ thì Model có kết quả chính xác cao
# Thế Nhưng Quy luật này sẽ không còn đúng với tập dữ liệu mới => Kết quả dự đoán chính xác thấp
# Method: Tách 1 phần dữ liệu ra khỏi quá trình huấn luyện rồi dùng phần đó để kiểm tra độ chính xác đối với dữ liệu mới
# Tập dữ liệu mới gọi là tập dữ liệu kiểm chứng (Validation Data)

# Triển khai
# Thư viện Scikit-Learn có một hàm tên là train_test_split
# Sử dụng dữ liệu đó để huấn luyện và dữ liệu còn lại thì kiểm chứng (validation) => Tính MAE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Random_state = 0 Lấy mặc định 75% train 25% validation
# Có thể chỉnh thành

# train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=1)
# Random_state = 1 giữ cố định
# test_size = 0.2 validation lấy 20%, traning = 80%
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

#Under Fitting and Overfitting
# MAE đã có phương pháp đáng tin cậy để đo lường độ chính xác của mô hình, Có thể thử nghiệm với các mô hình khác nhau để xem
# Mô hình nào đưa ra lựa chọn tốt nhất
# Underfitting: Mô hình học chưa đủ, dự đoán kém số ->node ít
# Overfitting: Mô hình học quá sát dữ liệu huấn luyện, kể cả những chi tiết riêng biệt hoặc nhiễu, nên dự đoán kém trên 
# dữ liệu mới, -> số node nhiều
# Vậy để không bị overfitting hay underfitting 
# Phải kiểm tra
# MAE Training cao + MAE Validation Cao = Underfitting
# MAE Traning thấp + MAE Validation Cao = Overfitting
# MAE Traning thấp + MAE Validation thấp = GOAT
# Cách để không bị Overfitting và Underfitting
# `max_leaf_nodes` Kiểm soát & cân bằng node lá tránh model bị 2 TH trên
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
# Hàm trên kiểm soát số Node tính MAE
# Sau khi tính được rồi thì
# Nên dùng vòng lặp để xem thử SO SÁNH ĐỘ CHÍNH XÁC của các mô hình của những giá trị MAX_NODES_LEAF khác nhau
# compare MAE with differing values of max_leaf_nodes
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
save_max_leaf_nodes = {}
for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    save_max_leaf_nodes[max_leaf_nodes] = my_mae

# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = min(save_max_leaf_nodes, key=save_max_leaf_nodes.get)
# Khi tìm được best rồi thì sẽ tổng kết lại model-> cải thiện model
from sklearn.tree import DecisionTreeRegressor

# Khởi tạo model với 50 leaf nodes
final_model = DecisionTreeRegressor(
    max_leaf_nodes=50,
    random_state=1
)

# Huấn luyện model bằng toàn bộ dữ liệu
final_model.fit(X, y)

# Hoàn thành được model nhưng DECISION TREE là model không quá tính vi -> HỌC Random Forest
# Có nhiều model khác tránh underfitting và overfitting hiệu quả hơn là -> Random Forest
# Cách Build Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))