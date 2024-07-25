import os

def main():
    input_directory = ".\\Add"  # 替换成包含数据文件的目录路径
    type1_directory = ".\\Type1"
    type2_directory = ".\\Type2"
    
    for filename in os.listdir(input_directory):
        input_string = os.path.join(input_directory, filename)
        if "I类" in input_string:
            output_string = os.path.join(type1_directory, filename)
            output_string = output_string.replace("I类-", "")
        if "II类" in input_string:
            output_string = os.path.join(type2_directory, filename)
            output_string = output_string.replace("II类-", "")
        
        os.rename(input_string, output_string)
        print(f"文件从 {input_string} 移动到 {output_string} 成功。")

if __name__ == "__main__":
    main()
