import os
import shutil

def main():
    input_directory = "/mnt/d/EE/project/windpowerforecast/data/实时预报"  
    type2_directory = "data/HistoricalWeatherForecastData/Type2"
    
    for filename in os.listdir(input_directory):
        input_string = os.path.join(input_directory, filename)
        output_string = os.path.join(type2_directory, filename)
        output_string = output_string.replace("[气象预测文件]II类-", "")
        
        shutil.copy(input_string, output_string)
        print(f"文件从 {input_string} 复制到 {output_string} 成功。")

if __name__ == "__main__":
    main()