#include <vector>
#include <iostream>
#include <algorithm>
#include <set>
#include <sstream>
#include <unordered_map>
#include <map>
#include <functional>
#include <unordered_set>
#include <stack>
#include <set>
#include <queue>
#define null INT16_MIN
using namespace std;

int multi_p_x(vector<int>& nums, int lft, int rht, int x) {
    if (lft >= rht) return x - nums[(lft + rht) / 2];
    else return multi_p_x(nums, lft, (lft + rht) / 2, x) * multi_p_x(nums, (lft + rht) / 2 + 1, rht, x);
}

double myPow(double x, int n) {
    if (n == 0) return 1;
    if (x == 0) return 0;
    long long m = n;
    if (m < 0) {
        x = 1 / x;
        m = -m;
    }
    while (m > 0) {
        if (m % 2 == 1) return x * myPow(x * x, m / 2);
        else return myPow(x * x, m / 2);
    }
    return 1;
}

int binarySearch_sqrt(vector<int>& nums, int target, bool flag) {
    int lft = 0, rht = nums.size() - 1, res = nums.size();
    while (lft <= rht) {
        int mid = (lft + rht) / 2;
        if (nums[mid] > target || (flag && nums[mid] >= target)) {
            rht = mid - 1;
            res = mid;
        }
        else {
            lft = mid + 1;
        }
    }
    return res;
}

int mySqrt(int x) {
    if (x == 0) return 0;
    int lft = 0, rht = x / 2 + 1;
    while (lft <= rht) {
        int mid = (lft + rht) / 2;
        long long r = mid * mid;
        if (r == x) return mid;
        else if (r < x) lft = mid + 1;
        else rht = mid - 1;
    }
    return rht;
}

bool check_peak(vector<int>& nums, int index) {
    int n = nums.size();
    if (index == 0) return nums[0] > nums[1];
    if (index == n - 1) return nums[n - 1] > nums[n - 2];
    return nums[index] > nums[index - 1] && nums[index] > nums[index + 1];
}

int findPeakElement(vector<int>& nums) {
    if (nums.size() == 1) return 0;
    int lft = 0, rht = nums.size() - 1;
    while (lft <= rht) {
        int mid = (lft + rht) / 2;
        if (check_peak(nums, mid)) return mid;
        if (mid + 1 < nums.size() && nums[mid] < nums[mid + 1]) lft = mid + 1;
        else rht = mid - 1;
    }
    return lft;
}

int singleNonDuplicate(vector<int>& nums) {
    //给定输入数组必然有奇数个元素, 不断二分查找, 对找到的元素对左右两侧, 奇数个的一侧肯定有要找的元素
    if (nums.size() == 1) return nums[0];
    int l = 0, r = nums.size() - 1, mid;
    int couple_index;
    while (l <= r) {
        if (r == l) return nums[l];
        mid = (l + r) / 2;
        if (nums[mid] == nums[mid - 1]) {
            if ((mid - 1 - l) % 2 == 1) r = max(l, mid - 2);
            else l = mid + 1;
        }
        else if (nums[mid] == nums[mid + 1]) {
            if ((mid - l) % 2 == 1) r = mid - 1;
            else l = min(mid + 2, r);
        }
        else return nums[mid];
    }
    return nums[0];
}

int main() {
    //a_plus_b();
    //cout << longestPalindromeSubseq("xaabacxcabaaxcabaax");
    vector <int> test = { 1,1,2,2,4,4,5,5,9};
    //cout << jump(test);
    vector<vector<char>> grid = { {'1','1','1','1','0'},{'1','1','0','1','0'},{'1','1','0','0','0'},{'0','0','0','0','0'} };
    vector <int> test2 = { 1,3,4,5,4 };
    vector<vector<int>> test3 = { {0,1} };
    vector<string> strs = { "eat","tea","tan","ate","nat","bat" };
    //cout << multi_p_x(test, 0, test.size() - 1, 22);
    //cout << myPow(2.0, -2147483648);
    //cout << mySqrt(10);
    //cout << findPeakElement(test2);
    cout << singleNonDuplicate(test);





}