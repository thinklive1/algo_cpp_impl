def find_medium(X, Y, l1, r1, l2, r2):  # 初始时l1=0,l2=0,r1=X.size()-1,r2=Y.size()-1
    if l1 == r1:# 退出条件： X和Y只剩一个元素，任取一个都是中位数
        return X[l1]
    x1 = (l1 + r1) // 2
    x2 = (l2 + r2) // 2
    m1 = X[x1]
    m2 = Y[x2]
    if m1 == m2:
        return m1
    elif m1 < m2:
        if (r1 - x1 - x2 + l2) > 0: #确保两个子数组长度相同
            x1 += 1
        return find_medium(X, Y, x1, r1, l2, x2)
    else:
        if (r2 - x2 - x1 + l1) > 0:
            x2 += 1
        return find_medium(X, Y, l1, x1, x2, r2)

print(find_medium([1, 9, 15, 32,33], [2, 8, 14, 21,44], 0, 4, 0, 4))
