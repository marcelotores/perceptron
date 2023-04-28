from ydata_profiling import ProfileReport
import ut

data_df = ut.im_data(10, data_frame=True)
print(data_df)

#profile = ProfileReport(data_df)

#profile

#profile.to_file("your_report.html")