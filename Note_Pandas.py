#Thư viện PANDAS
# Đọc dữ liệu excel thô sang code
import pandas as pd
#Có 2 core trong PANDAS
#DataFrames và Series
# Simple Data Frames
i =pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})
h =pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']})
#Sử dụng pd.DataFrame để tạo dữ liệu, key là các cột 'Bob', 'Sue', Còn value là các đối tượng còn lại
#Có thể đổi tên hàng bằng
#k =pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']},index=['Product A', 'Product B'])
#k =print(k)
# Data frames là 1 bảng còn series là một danh sách
series = pd.Series([1, 2, 3, 4, 5])
#print(series)
# Series chính là 1 cột trong data frame
k =pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')
print(k)
# Data frame thực chất là tổ hợp của các series gộp lại
# Việc tạo ra cũng được nhưng Pandas luôn làm với các dữ liệu có sẵn là tệp CSV
#Comma-Separated Values" (các giá trị được phân tách bằng dấu phẩy), viết tắt là CSV.
# CSV nạp vào Data Frame
# Sử dụng hàm pd.read_csv() để đọc dữ liệu data frames
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")
# DÙng .head() chỉ in ra 5 hàng đầu tiên
wine_reviews.head()
# Sử dụng cột làm chỉ số dùng
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wine_reviews.head()
# Cách lưu thành file CSV
wine_reviews.to_csv('toi bi gay.csv')

#INDEXING, SELECTING, ASSIGNING
# Chọn ra các dữ liệu cụ thể trong DataFrame và Series -> Học cách chọn lọc dữ liệu
# Có thể dễ dàng truy cập đọc dữ liệu trong cột
wine_reviews.country
# Chỉ in ra cột country
#Hoặc
wine_reviews['country']
# in ra đúng 1 phần tử
wine_reviews['country'][0]

#INDEXING PANDAS
# LOC & ILOC
# LOC theo nhãn
# data.loc[nhan_hang, nhan_cot]
# ILOC theo INDEX
#data.iloc[vi_tri_hang, vi_tri_cot]

# MANIPULATING INDEX
# Biến cột thành chỉ số index
wine_reviews.set_index("title")

# CONDITIONAL SLECTION
# Lọc dữ liệu 1 cách có điều kiện
# df[df['ten_cot'] dieu_kien]
wine_reviews.loc[wine_reviews.country == 'Italy']
wine_reviews.loc[(wine_reviews.country == 'Italy') & (wine_reviews.points >= 90)]
wine_reviews.loc[(wine_reviews.country == 'Italy') & (wine_reviews.points >= 90)]
# Chọn dữ liệu theo tổ hợp dùng isin
wine_reviews.loc[wine_reviews.country.isin(['Italy', 'France'])]
# Kiểm tra giá trị bị thiếu và không bị thiếu isnull() notnull()
wine_reviews.loc[wine_reviews.price.notnull()]

# Assigning data
# Là thay đổi giá trị cột của label đó hoặc thêm dữ liệu vào dataframe
wine_reviews['critic'] = 'everyone'
wine_reviews['index_backwards'] = range(len(wine_reviews), 0, -1)

# SUMARY FUNCTION and MAPS
# Định dạng dữ liệu
# SUMARY FUNCTION
# hàm describle()
# Thống kê tổng quan
#Thông số	Ý nghĩa
#count	Số giá trị không bị thiếu
#mean	Giá trị trung bình
#std	Độ lệch chuẩn
#min	Giá trị nhỏ nhất
#25%	Phân vị thứ 25
#50%	Trung vị
#75%	Phân vị thứ 75
#max	Giá trị lớn nhất
wine_reviews.describe()
# mean() tính giá trị trung bình
wine_reviews['points'].mean()
wine_reviews['price'].mean()
# unique() tìm các giá trị khác nhau
wine_reviews['country'].unique()
wine_reviews['points'].unique()
# count giá trị khác nhau
wine_reviews['country'].nunique()
# value_counts() — Đếm số lần xuất hiện
wine_reviews['country'].value_counts()
# MAP ( Ánh xạ)
# Đưa công thức -> kết quả 
review_points_mean = wine_reviews.points.mean()
wine_reviews.points.map(lambda p: p - review_points_mean)
# APPLY ( Biến đổi giá trị)
# MAP chỉ series, APPLY DATAFRAME VÀ SERIES
wine_reviews.apply(lambda col: col.mean())
# Grouping & sorting
# Từ ánh xạ hay apply() làm biến đổi dữ liệu. Chúng ta cần nhóm những dữ liệu đó lại, rồi thực hiện các thao tác cụ thể
# trên từng nhóm đó
# Thực hiện bằng hàm groupby(), hoặc hàm sắp xếp dữ liệu
wine_reviews.groupby('country').points.mean()
# lọc điểm trung bình từng quốc gia
# df.groupby('cột_nhóm').cột_cần_tính.hàm_tổng_hợp()
# result = reviews.groupby(['country', 'points']).price.mean()

#DATA types and MISSING DATA
# Xem kiểu dữ liệu từng lables
wine_reviews.dtypes
# Xem kiểu dữ liệu của 1 cột
wine_reviews['points'].dtype
# Đổi kiểu dữ liệu với astypes
wine_reviews['points'] = wine_reviews['points'].astype('int64')
# MISSING DATA - Giá trị bị thiếu
# Phát hiện giá trị bị thiếu
wine_reviews.isna()
# Tìm mỗi cột có bao nhiêu giá trị bị thiếu
wine_reviews.isna().sum()
# Ngược lại isna
wine_reviews.notna()
# Cách xử lý giá trị bị thiếu
# df.dropna()
# Xóa hàng giá trị bị thiếu
wine_reviews.dropna()
# Thay thế giá trị bị thiếu
wine_reviews.point.fillna(10)
wine_reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino")

# RENAMEING & COMNBINING
# Đổi tên & đổi cột
wine_reviews.rename(columns={'points': 'score'}) # ĐỔi label
wine_reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'}) # ĐỔi tên index đầu cột
# ĐỔi tên trục hàng và trục cột
wine_reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')
# Đổi tên đầu dòng wines trục hàng, fileds trục cột

#COMBINING
# GHép dữ liệu theo chiều dọc
canadian_youtube = pd.read_csv("../input/youtube-new/CAvideos.csv")
british_youtube = pd.read_csv("../input/youtube-new/GBvideos.csv")
pd.concat([canadian_youtube, british_youtube], axis= 0)
# Ghép dữu liệu theo chiều ngang
pd.concat([canadian_youtube, british_youtube], axis= 1)
