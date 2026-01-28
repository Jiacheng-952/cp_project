import random

def generate_fjspc_data(num_jobs, num_machines, avg_flex):
    output = []
    # 确保第一行能正确反映变量
    output.append(f"{num_jobs}\t{num_machines}\t{avg_flex}")

    print(f"开始生成 {num_jobs} 个工件...")
    for j in range(num_jobs):
        num_ops = random.randint(4, 7)
        job_line = [str(num_ops)+" "]
        for o in range(num_ops):
            # 这里的生成逻辑没问题
            k = random.randint(1, int(min(num_machines, avg_flex * 2 - 1)))
            job_line.append(str(k))
            selected_machines = random.sample(range(1, num_machines + 1), k)
            for m_id in selected_machines:
                job_line.append(str(m_id))
                job_line.append(str(random.randint(1, 10)))
        
        output.append(" ".join(job_line))
        # 调试打印：确认循环正在运行
        # print(f"已生成工件 {j+1}/{num_jobs}")

    print(f"正在生成 {num_machines}x{num_machines} 的 SDST 矩阵...")
    for i in range(num_machines):
        row = [str(0 if i == j else random.randint(1, 5)) for j in range(num_machines)]
        output.append(" ".join(row))

    return "\n".join(output)

for n in range(91, 101):  # 从1到100
    # 生成文件名，使用zfill确保3位数字
    filename = f"generate_MK{n:03d}.fjsc"
    filepath = f"fjspc/data/{filename}"
    
    print(f"正在生成第 {n} 个实例: {filename}")
    
    # 生成数据
    final_data = generate_fjspc_data(num_jobs=30, num_machines=20, avg_flex=3.5)
    
    # 将生成的数据保存到文件
    with open(filepath, "w") as f:
        f.write(final_data)

print(f"\n完成! 已生成100个实例文件，保存在 fjspc/data/ 目录下")