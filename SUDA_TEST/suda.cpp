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

template<typename container>
void prit_container(container c) {
    if (c.empty()) return;
    for (int i = 0;i < c.size();i++) cout << i << ": " << c[i] << '\n';
}

//c++输入输出,当cin>>从缓冲区中读取数据时，若缓冲区中第一个字符是空格、tab或换行这些分隔符时，cin>>会将其忽略并清除，继续读取下一个字符，若缓冲区为空，则继续等待。但是如果读取成功，字符后面的分隔符是残留在缓冲区的，cin>>不做
//1. 计算a+b
void a_plus_b() {
    cout << "please input two numbers : " << '\n';
    int a, b;
    while (cin >> a >> b) cout << "a+b is " << a + b << '\n';
}

//读取字符串并排序输出
void read_from_line() {
    string l;
    while (getline(cin, l)) {
        stringstream ss(l);
        int sum = 0, num;
        while (ss >> num) sum += num;
        cout << sum << '\n';
    }
}

void sort_strings() {
    string s;
    vector<string> vec;
    while (getline(cin, s)) {
        stringstream ss(s);//初始化语法
        s.clear();
        while (ss >> s) vec.push_back(s);
        sort(vec.begin(), vec.end());
        for (auto i = vec.begin();i != vec.end();i++) cout << *i << ' ';
        cout << '\n';
        vec.clear();
    }
}

void strings_withdot() {
    string s;
    while (getline(cin, s)) {
        stringstream ss(s);
        vector<string> vec;
        while (getline(ss, s, ',')) vec.push_back(s);//getline属于string头文件，可以指定分隔符 
        sort(vec.begin(), vec.end());
        for (int i = 0;i < vec.size() - 1;i++) cout << vec[i] << ',';
        cout << vec.back() << '\n';
    }
}

vector<int> twoSum(vector<int>& nums, int target) {// 暴力解法，O(N^2)
    vector<int> temp, result;
    for (auto i = nums.begin(); i != nums.end();i++) {
        temp.push_back(target - *i);
    }
    int index1 = 0;
    for (auto i = nums.begin(); i != nums.end();i++) {
        int index2 = index1 + 1;
        auto it = temp.begin();
        for (int t = 0;t < index2;t++) it++;
        while (it != temp.end()) {
            if (*i == *it) {
                result.push_back(index1);
                result.push_back(index2);
                return result;
            }
            it++;
            index2++;
        }
        index1++;
    }
    return result;
};

vector<int> twoSum2(vector<int>& nums, int target) {
    /*
    对数组的每个元素：
        1. 查找哈希表是否有其元素值对应的<k,v>,如果有，该<k,v>的v值就是其匹配的数组索引号
        2. 将target与该元素值的差作为k，其索引作为v写入哈希表
    */
    unordered_map<int, int> hash;
    for (int i = 0; i < nums.size();i++) {
        if (hash.find(nums[i]) != hash.end()) return { i,hash[nums[i]] };
        hash[target - nums[i]] = i;
    }
    return {};
}

//给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。必须在不复制数组的情况下原地对数组进行操作。
void moveZeroes(vector<int>& nums) {
    int n = nums.size(), slow = 0, fast = 0;
    if (n <= 1) return;
    while (fast < n) {
        if (nums[slow] == 0) {
            while (fast < n && nums[fast] == 0) ++fast;
            if (fast >= n) return;
            nums[slow] = nums[fast];
            nums[fast] = 0;
            ++slow;
        }
        else { ++slow;++fast; }
    }
}

int maxArea(vector<int>& height) {
    int i = 0, j = height.size() - 1, max_ar = (j - i) * min(height[i], height[j]);
    while (i < j) {
        if (height[i] < height[j]) ++i;
        else --j;
        max_ar = max(max_ar, (j - i) * min(height[i], height[j]));
    }
    return max_ar;
}

//一个非空整数数组 nums ，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素
int singleNumber(vector<int>& nums) {
    int temp = 0;
    for (int i : nums) temp ^= i;
    return temp;
}

struct ListNode {
    int val;
    ListNode* next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode* next) : val(x), next(next) {}
};

class Node {
public:
    int val;
    Node* next;
    Node* random;

    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};

Node* build_random_node(vector<vector<int>> vec) {
    vector<Node*> nodes;
    for (auto row : vec) {
        Node* n = new Node(row[0]);
        nodes.emplace_back(n);
    }
    for (int i = 0;i < vec.size();i++) {
        if (i != vec.size() - 1) nodes[i]->next = nodes[i + 1];
        else nodes[i]->next = nullptr;
        nodes[i]->random = vec[i][1] == 65536 ? nullptr : nodes[vec[i][1]];
    }
    return nodes[0];
}

//寻找链表公共节点
ListNode* getIntersectionNode(ListNode* headA, ListNode* headB) {
    int m = 0, n = 0;
    ListNode* t = headA->next;
    while (t != NULL) {
        ++m;
        t = t->next;
    }
    t = headB->next;
    while (t != NULL) {
        ++n;
        t = t->next;
    }
    if (m > n) for (int i = 0;i < m - n;i++) headA = headA->next;
    else for (int i = 0;i < n - m;i++) headB = headB->next;
    while (headA && headA != headB) { headA = headA->next;headB = headB->next; }
    return headA;
}

ListNode* getIntersectionNode_better(ListNode* headA, ListNode* headB) {//相当于同时遍历A+B和B+A
    if (headA == NULL || headB == NULL) {
        return NULL;
    }
    ListNode* pA = headA;
    ListNode* pB = headB;
    while (pA != pB) {
        pA = (pA == NULL ? headB : pA->next);
        pB = (pB == NULL ? headA : pB->next);
    }
    return pA;
}


ListNode* build_nodelist_withhead(ListNode* L, vector<int> lst) {
    ListNode* r = L;
    for (int i = 0; i < lst.size(); i++) {
        ListNode* s = new ListNode(lst[i]); // 创建新结点
        r->next = s;
        r = s;
    }
    r->next = NULL;
    return L;
}

ListNode* build_nodelist_withouthead(ListNode* L, vector<int> lst) {
    ListNode* r = L;
    r->val = lst[0];
    for (int i = 1; i < lst.size(); i++) {
        ListNode* s = new ListNode(lst[i]); // 创建新结点
        r->next = s;
        r = s;
    }
    r->next = NULL;
    return L;
}

ListNode* reverseList(ListNode* head) {
    if (!head || !head->next) return head;
    ListNode* p = head, * q = head->next;
    p->next = nullptr;
    while (q) {
        auto t = q->next;
        q->next = p;
        p = q;
        q = t;
    }
    return p;
}

//判断该链表是否为回文链表
bool isPalindrome(ListNode* head) {
    ListNode* slow = head, * fast = head;
    while (fast) {
        slow = slow->next;
        fast = fast->next;
        if (!fast) break;
        fast = fast->next;
    }
    slow = reverseList(slow);
    while (head && slow) {
        if (head->val != slow->val) return 0;
        head = head->next;
        slow = slow->next;
    }
    return 1;
}

bool hasCycle(ListNode* head) {
    ListNode* slow = head, * fast = head;
    while (fast) {
        if (slow == fast) return 1;
        slow = slow->next;
        fast = fast->next;
        if (!fast) return 0;
    }
    return 0;
}

//合并两个有序链表
ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
    ListNode* head = new ListNode{}; //声明一个新指针不会调用默认构造函数
    ListNode* tail = head, * t;
    while (list1 && list2) {
        t = list1->val < list2->val ? list1 : list2;
        if (t == list1) list1 = list1->next;
        else list2 = list2->next;
        tail->next = t;
        tail = tail->next;
    }
    tail->next = (!list1) ? list2 : list1;
    return head->next;
}

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode* left, TreeNode* right) : val(x), left(left), right(right) {}
};

using LinkTree = TreeNode*;

/* 获取索引为 i 节点的左子节点的索引 */
int left(int i) {
    return 2 * i + 1;
}

/* 获取索引为 i 节点的右子节点的索引 */
int right(int i) {
    return 2 * i + 2;
}

//从数组构建链式树，如果值是INT_MIN则视为nullptr
LinkTree build_tree_from_array(LinkTree T, vector<int> values, int start = 0) {
    int size = values.size();
    if (values[start] == INT16_MIN) return nullptr;
    T->val = values[start];
    T->left = (left(start) < size) ? build_tree_from_array(new TreeNode, values, left((start))) : nullptr;
    T->right = (right(start) < size) ? build_tree_from_array(new TreeNode, values, right((start))) : nullptr;
    return T;
}

vector<int> inorderTraversal(TreeNode* root) {
    if (!root) return {};
    vector<int> res;
    stack<TreeNode*> st;
    while (root || !st.empty()) {
        if (root) {
            st.emplace(root);
            root = root->left;
        }
        else {
            root = st.top();st.pop();
            res.push_back(root->val);
            root = root->right;
        }
    }
    return res;
}

//假设当前遍历到的节点为 x，将 x 的左子树中最右边的节点的右孩子指向 x，这样在左子树遍历完成后我们通过这个指向走回了 x，且能通过这个指向知晓我们已经遍历完成了左子树，而不用再通过栈来维护
vector<int> inorderTraversal_best(TreeNode* root) {
    vector<int> res;
    TreeNode* predecessor = nullptr;

    while (root != nullptr) {
        if (root->left != nullptr) {
            // predecessor 节点就是当前 root 节点向左走一步，然后一直向右走至无法走为止
            predecessor = root->left;
            while (predecessor->right != nullptr && predecessor->right != root) {
                predecessor = predecessor->right;
            }

            // 让 predecessor 的右指针指向 root，继续遍历左子树
            if (predecessor->right == nullptr) {
                predecessor->right = root;
                root = root->left;
            }
            // 说明左子树已经访问完了，我们需要断开链接
            else {
                res.push_back(root->val);
                predecessor->right = nullptr;
                root = root->right;
            }
        }
        // 如果没有左孩子，则直接访问右孩子
        else {
            res.push_back(root->val);
            root = root->right;
        }
    }
    return res;
}

int maxDepth(TreeNode* root) {
    queue<TreeNode*> qe1, qe2;
    qe1.emplace(root);
    int dep = 0, signal = 0;
    while (1) {
        if (qe1.empty()) return dep;
        while (!qe1.empty()) {
            if (qe1.front()->left) qe2.push(qe1.front()->left);
            if (qe1.front()->right) qe2.push(qe1.front()->right);
            qe1.pop();
        }
        ++dep;
        swap(qe1, qe2);
    }
}

//交换给定节点的左右子树
LinkTree& exchange_lr(LinkTree& root) {
    if (!root) return root;
    queue<TreeNode*> qe;
    qe.push(root);
    while (!qe.empty()) {
        TreeNode* t = qe.front();qe.pop();
        if (t->left) qe.push(t->left);
        if (t->right) qe.push(t->right);
        swap(t->left, t->right);
    }
    return root;
}

TreeNode* invertTree(TreeNode* root) {
    if (!root) return root;
    queue<TreeNode*> qe;
    qe.push(root);
    while (!qe.empty()) {
        TreeNode* t = qe.front();qe.pop();
        if (t->left) qe.push(t->left);
        if (t->right) qe.push(t->right);
        swap(t->left, t->right);
    }
    return root;
}

TreeNode* invertTree2(TreeNode* root) {
    if (!root) {
        return nullptr;
    }
    TreeNode* left = invertTree2(root->left);
    TreeNode* right = invertTree2(root->right);
    root->left = right;
    root->right = left;
    return root;
}

void invert_helper(TreeNode* node) {
    if (!node) return;
    swap(node->left, node->right);
    invert_helper(node->left);
    invert_helper(node->right);
}

TreeNode* invertTree3(TreeNode* root) {
    invert_helper(root);
    return root;
}

bool isCenterSymmetric(TreeNode* root) {
    if (!root) return 1;
    if (root->left && root->right && root->left->val == root->right->val) return isCenterSymmetric(root->left) && isCenterSymmetric(root->right);
    else if (!root->left && !root->right) return 1;
    return 0;
}

bool isSymmetric(TreeNode* root) {
    if (!root) return 1;
    stack<TreeNode*> st1, st2;
    TreeNode* p1 = root;
    TreeNode* p2 = p1;
    st1.emplace(p1);
    st2.emplace(p2);
    while (!st1.empty()) {
        p1 = st1.top();st1.pop();
        p2 = st2.top();st2.pop();
        if ((p1 && !p2) || (p2 && !p1)) return 0;
        else if (!p1) continue;
        else if (p1->val != p2->val) return 0;
        else {
            st1.push(p1->left);
            st1.push(p1->right);
            st2.push(p2->right);
            st2.push(p2->left);
        }
    }
    return 1;
}

bool check(TreeNode* p, TreeNode* q) {
    if (!p && !q) return true;
    if (!p || !q) return false;
    return p->val == q->val && check(p->left, q->right) && check(p->right, q->left);
}

bool isSymmetric2(TreeNode* root) {
    if (!root) return 1;
    return check(root->left, root->right);
}

int node_height(TreeNode* t) {
    if (!t)
        return 0;
    else {
        int temp = 1 + max(node_height(t->left), node_height(t->right));
        return temp;
    }
}

void diameterOfBinaryTree_helper(TreeNode* root, int& m) {
    if (!root) return;
    diameterOfBinaryTree_helper(root->left, m);
    diameterOfBinaryTree_helper(root->right, m);
    m = max(m, node_height(root->right) + node_height(root->left));
}

//二叉树的 直径 是指树中任意两个节点之间最长路径的 长度 。这条路径可能经过也可能不经过根节点
int diameterOfBinaryTree(TreeNode* root) {
    int dia = 0;
    diameterOfBinaryTree_helper(root, dia);
    return dia;
}

TreeNode* sortedArrayToBST_helper(vector<int>& nums, int head, int tail, TreeNode* parent, bool lft) {
    if (head > tail) return nullptr;
    if (head == tail) return new TreeNode(nums[head]);
    TreeNode* node = new TreeNode;
    int mid = (tail + head) / 2;
    node->val = nums[mid];
    node->left = sortedArrayToBST_helper(nums, head, mid - 1, parent, 1);
    node->right = sortedArrayToBST_helper(nums, mid + 1, tail, parent, 0);
    return node;
}

TreeNode* sortedArrayToBST(vector<int>& nums) {
    if (nums.empty()) return nullptr;
    int mid = nums.size() / 2;
    TreeNode* root = new TreeNode(nums[mid]);
    root->left = sortedArrayToBST_helper(nums, 0, mid - 1, root, 1);
    root->right = sortedArrayToBST_helper(nums, mid + 1, nums.size() - 1, root, 0);
    return root;
}

//给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置
int searchInsert(vector<int>& nums, int target) {
    int head = 0, tail = nums.size() - 1;
    while (head <= tail) {
        int mid = (head + tail) / 2;
        if (nums[mid] == target) return mid;
        else if (nums[mid] < target) head = mid + 1;
        else tail = mid - 1;
    }
    return head;
}

//判断括号字符串是否有效
bool isValid(string s) {
    stack<char> st;
    set<char> se = { '(','{','[' };
    if (se.find(s[0]) == se.end()) return 0;
    unordered_map<char, char> ma = { {')','('},{']','['},{'}','{'} };
    for (char c : s) {
        if (se.find(c) != se.end()) st.push(c);//左括号直接入栈
        else {//c是右括号
            if (st.empty() || ma[c] != st.top()) return 0;
            else st.pop();
        }
    }
    return st.empty();
}

int majorityElement(vector<int>& nums) {
    vector<int> times;
    return 0;
}



int maxProfit(vector<int>& prices) {
    int prev_min = INT16_MAX, pro = 0;;
    for (int i : prices) {
        prev_min = min(i, prev_min);
        pro = max(pro, i - prev_min);
    }
    return pro;
}

int climbStairs(int n) {
    if (n <= 2) return n;
    int steps = 2;
    vector<int> memory = { 0,1,2 };
    while (steps < n) {
        memory[0] = memory[1] + memory[2];
        memory[1] = memory[0];
        swap(memory[1], memory[2]);
        steps++;
    }
    return memory[0];
}

vector<vector<string>> groupAnagrams(vector<string>& strs) {
    unordered_map<string, vector<string>> um;
    vector<vector<string>> res;
    for (auto s : strs) {
        string t = s;
        sort(t.begin(), t.end());
        um[t].emplace_back(s);
    }
    for (auto i : um) res.push_back(i.second);
    return res;
}

int longestConsecutive(vector<int>& nums) {
    if (nums.empty()) return 0;
    unordered_set<int> us;
    int max_len = 0;
    for (auto i : nums) us.emplace(i);
    while (!us.empty()) {
        int i = *us.begin();
        int len = 0;
        while (us.count(i + 1)) ++i;
        while (us.count(i)) {
            ++len;
            us.erase(i);
            --i;
        }
        max_len = max(max_len, len);
    }
    return max_len;
}

int maxArea2(vector<int>& height) {
    int i = 0, j = height.size() - 1, max_ar = (j - i) * min(height[i], height[j]);
    while (i < j) {
        if (height[i] < height[j]) ++i;
        else --j;
        max_ar = max(max_ar, (j - i) * min(height[i], height[j]));
    }
    return max_ar;
}

//找到零和不重复的3数
/*需要保证：
    第二重循环枚举到的元素不小于当前第一重循环枚举到的元素；
    第三重循环枚举到的元素不小于当前第二重循环枚举到的元素。
    将数组中的元素从小到大进行排序，然后固定一个元素随后用双指针逼近另外两个
*/
vector<vector<int>> threeSum(vector<int>& nums) {
    if (nums.size() <= 2) return {};
    sort(nums.begin(), nums.end());
    vector<vector<int>> res;
    for (int i = 0;i < nums.size();i++) {
        if (i > 0 && nums[i] == nums[i - 1]) continue;
        int k = nums.size() - 1;
        for (int j = i + 1;j < nums.size();j++) {
            if (j > i + 1 && nums[j] == nums[j - 1]) continue;
            int t = nums[i] + nums[j];
            while (k > j && t + nums[k] > 0) --k;
            if (j == k) continue;
            if (nums[i] + nums[j] + nums[k] == 0) res.push_back({ nums[i], nums[j], nums[k] });
        }
    }
    return res;
}

int lengthOfLongestSubstring(string s) {
    if (s.size() <= 1) return s.size();
    unordered_set<char> hash;
    size_t max_len = 1;
    int i = 0, j = 1;
    hash.insert(s[i]);
    while (1) {
        while (j < s.size() && !hash.count(s[j])) {
            hash.insert(s[j]);
            ++j;
        }
        max_len = max(max_len, hash.size());
        if (j == s.size()) break;
        else {
            hash.erase(s[i]);
            ++i;
        }
    }
    return max_len;
}

vector<int> findAnagrams(string s, string p) {
    vector<int> res;
    unordered_map<char, int> hash_p;
    for (char c : p) {
        if (!hash_p.count(c)) hash_p[c] = 1;
        else ++hash_p[c];
    }
    int i = 0, j = 0;
    while (1) {
        auto hash = hash_p;
        int  j = i;
        while (j < s.size() && !hash.empty()) {//检查以i开始的子串是否匹配
            if (!hash.count(s[j])) break;
            else {
                --hash[s[j]];
                if (hash[s[j]] == 0) hash.erase(s[j]);
                ++j;
            }
        }
        if (hash.empty()) res.emplace_back(i);//如果匹配添加索引
        if (j >= s.size()) break;//如果越界退出
        //不匹配需要移到下一个可能匹配的i
        ++i;
        while (!hash_p.count(s[i])) {
            if (i > s.size() - p.size()) return res;
            ++i;
        }
    }
    return res;
}

vector<int> findAnagrams_with_window(string s, string p) {
    int sLen = s.size(), pLen = p.size();

    if (sLen < pLen) {
        return vector<int>();
    }

    vector<int> ans;
    vector<int> sCount(26);
    vector<int> pCount(26);
    for (int i = 0; i < pLen; ++i) {
        ++sCount[s[i] - 'a'];
        ++pCount[p[i] - 'a'];
    }

    if (sCount == pCount) {
        ans.emplace_back(0);
    }

    for (int i = 0; i < sLen - pLen; ++i) {
        --sCount[s[i] - 'a'];
        ++sCount[s[i + pLen] - 'a'];

        if (sCount == pCount) {
            ans.emplace_back(i + 1);
        }
    }

    return ans;
}

vector<int> findAnagrams_best(string s, string p) {//统计滑动窗口和字符串 p 中每种字母数量的差
    int sLen = s.size(), pLen = p.size();
    if (sLen < pLen) return {};
    vector<int> ans;
    vector<int> count(26);
    for (int i = 0; i < pLen; ++i) {//计算第一个窗口的字符情况
        ++count[s[i] - 'a'];
        --count[p[i] - 'a'];
    }
    int differ = 0;//子串和p不同的字母数量
    for (int j = 0; j < 26; ++j) if (count[j] != 0) ++differ;//不为0的元素说明有差异
    if (differ == 0) ans.emplace_back(0);
    for (int i = 0; i < sLen - pLen; ++i) {
        --count[s[i] - 'a'];//i右移移除对应count，正式的+1由for循环完成,接下来处理differ变化
        if (count[s[i] - 'a'] == 0) --differ;  // 如果i处是多余字符，即count的对应值为1，i右移后differ会减少
        else if (count[s[i] - 'a'] == -1)  ++differ; // 如果i处正好相抵，i右移后differ会增加

        ++count[s[i + pLen] - 'a'];
        if (count[s[i + pLen] - 'a'] == 0)  --differ;
        else if (count[s[i + pLen] - 'a'] == 1)  ++differ;

        if (differ == 0) ans.emplace_back(i + 1);
    }
    return ans;
}

//mp记录从0开始的子序列和及对应序列数量，若[0-j]-[0-i]==k，则找到了mp对应数量的子序列
int subarraySum(vector<int>& nums, int k) {
    unordered_map<int, int> mp;
    mp[0] = 1;
    int count = 0, pre = 0;
    for (auto& x : nums) {
        pre += x;
        if (mp.count(pre - k)) {
            count += mp[pre - k];
        }
        mp[pre]++;
    }
    return count;
}

int maxSubArray(vector<int>& nums) {
    unordered_map<int, int> mp;
    if (nums.size() == 1) return nums[0];
    int max_sum = nums[0], pre_sum = max_sum, min_sum = min(0, nums[0]);
    for (int i = 1;i < nums.size();i++) {
        min_sum = min(min_sum, pre_sum);
        pre_sum += nums[i];
        max_sum = max(max_sum, pre_sum - min_sum);
    }
    return max_sum;
}

/*
线段树
 struct Status {
        int lSum, rSum, mSum, iSum;
    };

    Status pushUp(Status l, Status r) {
        int iSum = l.iSum + r.iSum;
        int lSum = max(l.lSum, l.iSum + r.lSum);
        int rSum = max(r.rSum, r.iSum + l.rSum);
        int mSum = max(max(l.mSum, r.mSum), l.rSum + r.lSum);
        return (Status) {lSum, rSum, mSum, iSum};
    };

    Status get(vector<int> &a, int l, int r) {
        if (l == r) {
            return (Status) {a[l], a[l], a[l], a[l]};
        }
        int m = (l + r) >> 1;
        Status lSub = get(a, l, m);
        Status rSub = get(a, m + 1, r);
        return pushUp(lSub, rSub);
    }

    int maxSubArray(vector<int>& nums) {
        return get(nums, 0, nums.size() - 1).mSum;
    }

*/

vector<vector<int>> merge_intervals(vector<vector<int>>& intervals) {
    if (intervals.empty()) return {};
    if (intervals.size() == 1) return intervals;
    sort(intervals.begin(), intervals.end());
    vector<vector<int>> res = { intervals[0] };
    for (int i = 1;i < intervals.size();i++) {
        int tail = res.size() - 1;
        if (intervals[i][0] <= res[tail][1]) {
            res[tail][1] = max(intervals[i][1], res[tail][1]);
        }
        else res.emplace_back(intervals[i]);
    }
    return res;
}

void reverse(vector<int>& nums, int start, int end) {
    for (int i = 0;i < (end - start + 1) / 2;i++) swap(nums[start + i], nums[end - i]);
}

void right_rotate_vector(vector<int>& nums, int k) {
    k %= nums.size();
    if (k == 0) return;
    reverse(nums, 0, nums.size() - k - 1);
    reverse(nums, nums.size() - k, nums.size() - 1);
    reverse(nums, 0, nums.size() - 1);
    prit_container(nums);
}

vector<int> productExceptSelf(vector<int>& nums) {
    const int size = nums.size();
    vector<int> res(nums.size());
    res[0] = 1;
    for (int i = 1;i < size;i++) res[i] = res[i - 1] * nums[i - 1];
    int right_product = nums[size - 1];
    for (int j = size - 2;j >= 0;--j) {
        res[j] *= right_product;
        right_product *= nums[j];
    }
    return res;
}

void setZeroes(vector<vector<int>>& matrix) {

}

/*

设链表中环外部分的长度为 a。slow 指针进入环后，又走了 b 的距离与 fast 相遇,到环前最后一个结点剩下距离为c。此时，fast 指针已经走完了环的 n 圈，因此它走过的总距离为 a+n(b+c)+b=a+(n+1)b+nc
即a+(n+1)b+nc=2(a+b)⟹a=c+(n−1)(b+c)
当发现 slow 与 fast 相遇时，我们再额外使用一个指针 ptr。起始，它指向链表头部；随后，它和 slow 每次向后移动一个位置。最终，它们会在入环点相遇
*/

ListNode* detectCycle(ListNode* head) {
    ListNode* slow = head, * fast = head;
    while (fast != nullptr) {
        slow = slow->next;
        fast = fast->next;
        if (fast == nullptr) return nullptr;
        fast = fast->next;
        if (fast == slow) {
            ListNode* ptr = head;
            while (ptr != slow) {
                ptr = ptr->next;
                slow = slow->next;
            }
            return ptr;
        }
    }
    return nullptr;
}

void print_linknode(ListNode* l) {
    int i = 0;
    while (l) {
        cout << i << ": " << l->val << '\n';
        ++i;
        l = l->next;
    }
}

ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
    ListNode* head = nullptr, * tail = nullptr;
    int carry = 0;
    while (l1 || l2) {
        int n1 = l1 ? l1->val : 0;
        int n2 = l2 ? l2->val : 0;
        int sum = n1 + n2 + carry;
        if (!head) {
            head = tail = new ListNode(sum % 10);
        }
        else {
            tail->next = new ListNode(sum % 10);
            tail = tail->next;
        }
        carry = sum / 10;
        if (l1) {
            l1 = l1->next;
        }
        if (l2) {
            l2 = l2->next;
        }
    }
    if (carry > 0) {
        tail->next = new ListNode(carry);
    }
    return head;
}

ListNode* removeNthFromEnd(ListNode* head, int n) {
    ListNode* fast = head, * slow = head, * pre = slow;
    for (int i = 0;i < n; i++) {
        fast = fast->next;
    }
    while (fast) {
        pre = slow;
        slow = slow->next;
        fast = fast->next;
    }
    if (pre == slow) return head->next;
    pre->next = slow->next;
    return head;
}

ListNode* swapPairs(ListNode* head) {//两两交换相邻的节点，并返回交换后链表的头节点
    ListNode* dummyHead;
    dummyHead->val = 0;
    dummyHead->next = head;
    ListNode* temp = dummyHead;
    while (temp->next != nullptr && temp->next->next != nullptr) {//之后两个节点都存在时才可以交换
        ListNode* node1 = temp->next;
        ListNode* node2 = temp->next->next;
        temp->next = node2;
        node1->next = node2->next;
        node2->next = node1;
        temp = node1;
    }
    ListNode* res = dummyHead->next;
    return res;
}

Node* copyRandomList(Node* head) {
    unordered_map<Node*, Node*> cachedNode;
    if (head == nullptr) return head;
    if (!cachedNode.count(head)) {
        Node* headNew = new Node(head->val);
        cachedNode[head] = headNew;
        headNew->next = copyRandomList(head->next);
        headNew->random = copyRandomList(head->random);
    }
    return cachedNode[head];
}

Node* copyRandomList2(Node* head) {
    unordered_map<Node*, Node*> cachedNode;
    if (head == nullptr) return head;
    Node* t = head;
    while (t) {
        Node* headNew = new Node(t->val);
        cachedNode[t] = headNew;
        t = t->next;
    }
    for (auto n : cachedNode) {
        Node* n0 = n.first;
        Node* n1 = n.second;
        n1->next = n0->next == nullptr ? nullptr : cachedNode[n0->next];
        n1->random = n0->random == nullptr ? nullptr : cachedNode[n0->random];
    }
    return cachedNode[head];
}

vector<vector<int>> levelOrder(TreeNode* root) {
    vector <vector <int>> ret;
    if (!root) {
        return ret;
    }
    queue <TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        int currentLevelSize = q.size();
        ret.push_back(vector <int>());
        for (int i = 1; i <= currentLevelSize; ++i) {
            auto node = q.front(); q.pop();
            ret.back().push_back(node->val);
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
    }
    return ret;
}

void isValidBST_helper(TreeNode* root, long long min_val, long long max_val, bool& res) {
    if (!res || !root) return;//检验失败或root为空则退出
    if (min_val >= root->val || root->val >= max_val) { res = 0;  return; }
    isValidBST_helper(root->left, min_val, root->val, res);
    isValidBST_helper(root->right, root->val, max_val, res);
}
bool isValidBST(TreeNode* roots) {
    bool i = 1;
    isValidBST_helper(roots, INT32_MIN, INT32_MAX, i);
    return i;
}

int node_num_of_tree(TreeNode* root) {
    if (!root) return 0;
    else return 1 + node_num_of_tree(root->left) + node_num_of_tree(root->right);
}

int kthSmallest_helper(TreeNode* root, const int& k, int n) {//n表示root这个节点的大小顺序
    if (!root) return 0;
    if (n == k) return root->val;
    else if (n < k) return kthSmallest_helper(root->right, k, node_num_of_tree(root->right->left) + n + 1);//由于题目必然有解，此时root必存在右孩子
    else return kthSmallest_helper(root->left, k, n - node_num_of_tree(root->left->right) - 1);
}

int kthSmallest(TreeNode* root, int k) {
    return kthSmallest_helper(root, k, 1 + node_num_of_tree(root->left));
}

vector<int> rightSideView(TreeNode* root) {
    if (!root) return {};
    queue<TreeNode*> qe1, qe2;
    qe1.emplace(root);
    vector<int> res = { root->val };
    while (1) {
        while (!qe1.empty()) {
            auto t = qe1.front();
            qe1.pop();
            if (t->left) qe2.emplace(t->left);
            if (t->right) qe2.emplace(t->right);
        }
        if (qe2.empty()) return res;
        res.emplace_back(qe2.back()->val);
        swap(qe1, qe2);
    }
    return res;
}

//对于当前节点，如果其左子节点不为空，则在其左子树中找到最右边的节点，作为前驱节点，将当前节点的右子节点赋给前驱节点的右子节点，然后将当前节点的左子节点赋给当前节点的右子节点，并将当前节点的左子节点设为空。对当前节点处理结束后，继续处理链表中的下一个节点，直到所有节点都处理结束
void flatten_best(TreeNode* root) {
    TreeNode* curr = root;
    while (curr != nullptr) {
        if (curr->left != nullptr) {
            TreeNode* next = curr->left;
            TreeNode* predecessor = next;
            while (predecessor->right != nullptr) {
                predecessor = predecessor->right;
            }
            predecessor->right = curr->right;//对一个节点来说，其右孩子先序遍历的前驱是左孩子最右的子节点
            curr->left = nullptr;
            curr->right = next;
        }
        curr = curr->right;
    }
}

//维护上一个访问的节点prev，每次访问一个节点时，令当前访问的节点为curr，将prev的左子节点设为null以及将prev的右子节点设为curr，然后将curr赋值给prev，进入下一个节点的访问，直到遍历结束。需要注意的是，初始时prev为null，只有在prev不为null时才能对prev的左右子节点进行更新
void flatten(TreeNode* root) {
    if (root == nullptr) {
        return;
    }
    auto stk = stack<TreeNode*>();
    stk.push(root);
    TreeNode* prev = nullptr;
    while (!stk.empty()) {
        TreeNode* curr = stk.top(); stk.pop();
        if (prev != nullptr) {
            prev->left = nullptr;
            prev->right = curr;
        }
        TreeNode* left = curr->left, * right = curr->right;
        if (right != nullptr) stk.push(right);
        if (left != nullptr) stk.push(left);
        prev = curr;
    }
}


TreeNode* BuildTree_helper(const vector<int>& preorder, const vector<int>& inorder, int preorder_left, int preorder_right, int inorder_left, int inorder_right, const unordered_map<int, int>& index) {
    if (preorder_left > preorder_right) return nullptr;
    // 前序遍历中的第一个节点就是根节点
    int preorder_root = preorder_left;
    // 在中序遍历中定位根节点
    int inorder_root = index.at(preorder[preorder_root]);
    TreeNode* root = new TreeNode(preorder[preorder_root]);
    int size_left_subtree = inorder_root - inorder_left;
    // 递归地构造左子树，并连接到根节点
    // 先序遍历中「从 左边界+1 开始的 size_left_subtree」个元素就对应了中序遍历中「从 左边界 开始到 根节点定位-1」的元素
    root->left = BuildTree_helper(preorder, inorder, preorder_left + 1, preorder_left + size_left_subtree, inorder_left, inorder_root - 1, index);
    // 递归地构造右子树，并连接到根节点
    // 先序遍历中「从 左边界+1+左子树节点数目 开始到 右边界」的元素就对应了中序遍历中「从 根节点定位+1 到 右边界」的元素
    root->right = BuildTree_helper(preorder, inorder, preorder_left + size_left_subtree + 1, preorder_right, inorder_root + 1, inorder_right, index);
    return root;
}

TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
    unordered_map<int, int> index;
    int n = preorder.size();
    // 构造哈希映射，用于快速定位根节点
    for (int i = 0; i < n; ++i) {
        index[inorder[i]] = i;
    }
    return BuildTree_helper(preorder, inorder, 0, n - 1, 0, n - 1, index);
}


int pathSum_helper(TreeNode* root, long long curr, int targetSum, unordered_map<long long, int>& prefix) {
    if (!root) return 0;
    int ret = 0;
    curr += root->val;
    if (prefix.count(curr - targetSum)) ret = prefix[curr - targetSum];
    prefix[curr]++;
    ret += pathSum_helper(root->left, curr, targetSum, prefix);
    ret += pathSum_helper(root->right, curr, targetSum, prefix);
    prefix[curr]--;
    return ret;
}

int pathSum(TreeNode* root, int targetSum) {
    unordered_map<long long, int> prefix;
    prefix[0] = 1;
    return pathSum_helper(root, 0, targetSum, prefix);
}

void lowestCommonAncestor_helper(TreeNode* root, unordered_map<int, TreeNode*>& fa) {//构建子节点到父节点的链接
    if (root->left != nullptr) {
        fa[root->left->val] = root;
        lowestCommonAncestor_helper(root->left, fa);
    }
    if (root->right != nullptr) {
        fa[root->right->val] = root;
        lowestCommonAncestor_helper(root->right, fa);
    }
}
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    unordered_map<int, TreeNode*> fa;
    unordered_map<int, bool> vis;
    fa[root->val] = nullptr;
    lowestCommonAncestor_helper(root, fa);
    while (p != nullptr) {//一路标记p的父节点
        vis[p->val] = true;
        p = fa[p->val];
    }
    while (q != nullptr) {
        if (vis[q->val]) return q;
        q = fa[q->val];
    }
    return nullptr;
}

void permute_helper(vector<vector<int>>& res, vector<int>& output, int first, int len) {
    // 所有数都填完了
    if (first == len) {
        res.emplace_back(output);
        return;
    }
    for (int i = first; i < len; ++i) {//[0,first]部分为已填过的数
        // 动态维护数组
        swap(output[i], output[first]);
        // 继续递归填下一个数
        permute_helper(res, output, first + 1, len);
        // 撤销操作
        swap(output[i], output[first]);
    }
}

vector<vector<int>> permute(vector<int>& nums) {
    vector<vector<int> > res;
    permute_helper(res, nums, 0, (int)nums.size());
    return res;
}

vector<vector<int>> subsets(vector<int>& nums) {//全子集
    vector<int> t;
    vector<vector<int>> res;
    int n = nums.size();
    for (int mask = 0; mask < (1 << n); ++mask) {//n位二进制掩码表示子集
        t.clear();
        for (int i = 0; i < n; ++i) if (mask & (1 << i)) t.push_back(nums[i]);
        res.push_back(t);
    }
    return res;
}

void letterCombinations_helper(vector<string>& combinations, const unordered_map<char, string>& phoneMap, const string& digits, int index, string& combination) {
    if (index == digits.length()) {
        combinations.push_back(combination);
    }
    else {
        char digit = digits[index];
        const string& letters = phoneMap.at(digit);
        for (const char& letter : letters) {
            combination.push_back(letter);
            letterCombinations_helper(combinations, phoneMap, digits, index + 1, combination);
            combination.pop_back();
        }
    }
}
vector<string> letterCombinations(string digits) {//电话数字能表示的字母组合
    if (digits.empty()) return {};
    vector<string> combinations;
    unordered_map<char, string> phoneMap{
        {'2', "abc"},
        {'3', "def"},
        {'4', "ghi"},
        {'5', "jkl"},
        {'6', "mno"},
        {'7', "pqrs"},
        {'8', "tuv"},
        {'9', "wxyz"}
    };
    string combination;
    letterCombinations_helper(combinations, phoneMap, digits, 0, combination);
    return combinations;
}

//杨辉三角
vector<vector<int>> generate(int numRows) {
    if (numRows == 1) return { {1} };
    vector<vector<int>> res = { {1} };
    for (int i = 1;i <= numRows - 1;++i) {
        vector<int> row = { res[i - 1][0] };
        for (int j = 1;j < i;++j) row.emplace_back(res[i - 1][j - 1] + res[i - 1][j]);
        row.emplace_back(res[i - 1][i - 1]);
        res.emplace_back(row);
    }
    return res;
}

//沿街的房屋每间都藏有一定的现金,如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警,计算一夜之内能够偷窃到的最高金额
int rob(const vector<int>& nums) {
    int n = nums.size(), i = 2;
    if (n == 1) return nums[0];
    vector<int> mem = { max(nums[0], nums[1]),nums[0],max(nums[0], nums[1]) };
    while (i < n) {
        mem[0] = max(mem[1] + nums[i], mem[2]);
        mem[1] = mem[2];
        mem[2] = mem[0];
        ++i;
    }
    return mem[0];
}

void setZeroes2(vector<vector<int>>& matrix) {
    int m = matrix.size();
    int n = matrix[0].size();
    vector<int> row(m), col(n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (!matrix[i][j]) {
                row[i] = col[j] = true;
                break;
            }
        }
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (row[i] || col[j]) {
                matrix[i][j] = 0;
            }
        }
    }
}

vector<int> spiralOrder(vector<vector<int>>& matrix) {
    vector<int> res;
    int m = matrix.size();
    int n = matrix[0].size();
    int dire = 0, cycle = 0;
    while (1) {
        if (res.size() >= m * n) return res;
        switch (dire) {
        case 0:
        /* code */
        for (int j = cycle;j < n - cycle;++j) res.emplace_back(matrix[cycle][j]);
        dire = 1;
        break;
        case 1:
        for (int i = cycle + 1;i < m - cycle;++i) res.emplace_back(matrix[i][n - 1 - cycle]);
        dire = 2;
        break;
        case 2:
        for (int j = n - cycle - 2;j >= cycle;--j) res.emplace_back(matrix[m - 1 - cycle][j]);
        dire = 3;
        break;
        case 3:
        for (int i = m - 2 - cycle;i >= 1 + cycle;--i) res.emplace_back(matrix[i][cycle]);
        dire = 0;
        ++cycle;
        break;
        }
    }
    return res;

}

void rotate(vector<vector<int>>& matrix) {
    int n = matrix.size();
    for (int i = 0; i < n / 2; ++i) for (int j = 0; j < n; ++j) swap(matrix[i][j], matrix[n - i - 1][j]);
    for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) swap(matrix[i][j], matrix[j][i]);
}

bool searchMatrix(vector<vector<int>>& matrix, int target) {
    for (const auto& row : matrix) {
        auto it = lower_bound(row.begin(), row.end(), target);
        if (it != row.end() && *it == target) return true;
    }
    return false;
}

bool searchMatrix2(vector<vector<int>>& matrix, int target) {
    int m = matrix.size(), n = matrix[0].size();
    int x = 0, y = n - 1;
    while (x < m && y >= 0) {
        if (matrix[x][y] == target) {
            return true;
        }
        if (matrix[x][y] > target) {
            --y;
        }
        else {
            ++x;
        }
    }
    return false;
}

bool is_outside(int i, int j, const int& m, const int& n, vector<vector<char>>& grid) {
    return (i < 0 || i >= n || j < 0 || j >= m);
}

void grapf_bfs(vector<vector<char>>& grid, int i, int j, const int& m, const int& n) {
    if (is_outside(i, j, m, n, grid)) return;
    else if (grid[i][j] == '0') return;
}

int numIslands(vector<vector<char>>& grid) {
    int n = grid.size();//行数
    int m = grid[0].size();//列数
    int res = 0;
    queue<pair<int, int>> qe;
    for (int i = 0;i < n;++i) {
        for (int j = 0;j < m;++j) {
            if (grid[i][j] == '1') {
                qe.emplace(i, j);
                while (!qe.empty()) {
                    auto p = qe.front();qe.pop();
                    int ti = p.first, tj = p.second;
                    if (grid[ti][tj] == '0') continue;//遇到水终止操作
                    grid[ti][tj] = '2'; cout << i << ':' << j << '\n';
                    if (!is_outside(ti - 1, tj, m, n, grid) && grid[ti - 1][tj] != '2') { qe.emplace(ti - 1, tj); grid[ti - 1][tj] = '2'; }
                    if (!is_outside(ti + 1, tj, m, n, grid) && grid[ti + 1][tj] != '2') { qe.emplace(ti + 1, tj); grid[ti + 1][tj] = '2'; }
                    if (!is_outside(ti, tj - 1, m, n, grid) && grid[ti][tj - 1] != '2') { qe.emplace(ti, tj - 1); grid[ti][tj - 1] = '2'; }
                    if (!is_outside(ti, tj + 1, m, n, grid) && grid[ti][tj + 1] != '2') { qe.emplace(ti, tj + 1); grid[ti][tj + 1] = '2'; }
                }
                ++res;
            }
        }
    }
    return res;
}

int orangesRotting(vector<vector<int>>& grid) {
    queue<pair<int, int>> q;
    int dirs[4][2] = { {0,1},{0,-1},{1,0},{-1,0} };
    int m = grid.size(), n = grid[0].size();
    bool flag = false;//是否有新鲜橘子
    for (int i = 0;i < m;i++)
        for (int j = 0;j < n;j++)
            if (grid[i][j] == 2)//找到第一轮腐烂橘子
                q.push({ i,j });
            else if (grid[i][j] == 1)
                flag = 1;
    if (q.empty() && flag)//如果没有腐烂橘子并且有新鲜橘子，则新鲜橘子不可能腐烂
        return -1;
    int ans = 0;
    while (!q.empty()) {
        int t = q.size();//遍历同一时间感染的橘子
        for (int k = 0;k < t;k++) {
            pair<int, int>p = q.front(); q.pop();
            for (auto dir : dirs) {
                int x = p.first + dir[0];
                int y = p.second + dir[1];
                if (x >= 0 && x < m && y >= 0 && y < n && grid[x][y] == 1)//找到新感染的橘子
                {
                    grid[x][y] = 2;
                    q.push({ x,y });
                }
            }
        }
        if (!q.empty())//如果当前轮有新感染的橘子，时间加一
            ans++;
    }
    for (int i = 0;i < m;i++)
        for (int j = 0;j < n;j++)
            if (grid[i][j] == 1)//检查是否有未感染的新鲜橘子
                return -1;
    return ans;
}

void dfs_course(vector<int>& courses, const vector<vector<int>>& edges, int node, bool& res) {
    if (!res) return;
    courses[node] = 1;
    auto es = edges[node];//该节点连向的其他节点
    for (auto e : es) {
        if (courses[e] == 1) { res = 0; return; }
        else if (courses[e] == 0) dfs_course(courses, edges, e, res);
    }
    courses[node] = 2;
}

bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
    vector<int> courses(numCourses, 0);//课程状态数组，0表示未遍历过
    vector<vector<int>> edges(numCourses, vector<int>());
    bool res = 1;
    for (const auto& e : prerequisites) edges[e[1]].push_back(e[0]);
    for (int i = 0;i < numCourses && res;++i) if (courses[i] == 0) dfs_course(courses, edges, i, res);
    return res;
}

void combinationSum_helper(vector<vector<int>>& res, const vector<int> candidates, const int& target, int pre_sum, vector<int> pre_select, int index) {
    if (pre_sum > target) return;
    else if (pre_sum == target) {
        res.emplace_back(pre_select);
    }
    else {
        for (int i = index;i < candidates.size();++i) {
            int sum = pre_sum + candidates[i];
            if (sum > target) break;
            else {
                pre_select.emplace_back(candidates[i]);
                combinationSum_helper(res, candidates, target, sum, pre_select, i);
                pre_select.pop_back();
            }
        }
    }
}

vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
    vector<vector<int>> res;
    sort(candidates.begin(), candidates.end());
    int size = candidates.size();
    combinationSum_helper(res, candidates, target, 0, {}, 0);
    return res;
}

void find_symmetry(const string& s, vector<vector<uint8_t>>& sym_word, int i, int j);

void backtrack_string(const string& s, int i, vector<vector<string>>& res, vector<string> selection, vector<vector<uint8_t>>& sym_word) {
    int n = s.size();
    if (i == n) {
        res.emplace_back(selection);
        return;
    }
    for (int j = i; j < n; ++j) {
        find_symmetry(s, sym_word, i, j);
        if (sym_word[i][j] == 1) {
            selection.emplace_back(s.substr(i, j - i + 1));
            backtrack_string(s, j + 1, res, selection, sym_word);
            selection.pop_back();
        }
    }

}

void find_symmetry(const string& s, vector<vector<uint8_t>>& sym_word, int i, int j) {
    if (sym_word[i][j] != 0) return;
    if (i >= j) sym_word[i][j] = 1;
    else {
        find_symmetry(s, sym_word, i + 1, j - 1);
        sym_word[i][j] = (s[i] == s[j] && sym_word[i + 1][j - 1] == 1) ? 1 : 2;
    }
}

vector<vector<string>> partition(string s) {
    int n = s.length();
    vector<vector<uint8_t>> sym_word(n, vector<uint8_t>(n, 0));
    for (int i = 0; i < n;++i) {
        for (int j = i;j < n;++j) sym_word[i][j] = 1;
    }
    vector<vector<string>> res;
    vector<string> selection;
    find_symmetry(s, sym_word, 0, n - 1);
    backtrack_string(s, 0, res, selection, sym_word);
    return res;
}

int binarySearch(vector<int>& nums, int target, bool flag) {
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

pair<int, int> binarySearchv2(const vector<int>& nums, const int target) {
    int lft = 0, rht = nums.size() - 1, mid;
    while (lft <= rht) {
        //执行二分查找, 如果成功, mid 就是目标值的索引；如果失败, lft或者rht会正好是小于和大于target的数中最接近它的两个数的索引
        mid = (lft + rht) / 2;
        if (nums[mid] > target) {
            rht = mid - 1;
        }
        else {
            lft = mid + 1;
        }
    }
    if (nums[mid] == target) return { mid,mid };
    return { lft, rht };
}

int binarySearch_lft(const vector<int>& nums, int target) {
    int lft = 0, rht = nums.size() - 1, res;
    while (lft <= rht) {
        int mid = (lft + rht) / 2;
        if (nums[mid] >= target) {
            rht = mid - 1;
            res = mid;
        }
        else lft = mid + 1;
    }
    return res;
}

int binarySearch_rht(const vector<int>& nums, int target) {
    int lft = 0, rht = nums.size() - 1, res;
    while (lft <= rht) {
        int mid = (lft + rht) / 2;
        if (nums[mid] <= target) {
            lft = mid + 1;
            res = mid;
        }
        else rht = mid - 1;
    }
    return res;
}

vector<int> searchRange(vector<int>& nums, int target) {
    if (nums.size() == 0) return { -1,-1 };
    int lft = binarySearch_lft(nums, target);
    int rht = binarySearch_rht(nums, target);
    if (lft < 0 || rht >= nums.size() || nums[lft] != target || nums[rht] != target) lft = rht = -1;
    return { lft,rht };
}

bool is_sorted(const vector<int>& nums, int lft, int rht) {
    return lft == rht || nums[lft] < nums[rht];
}


int search_rotate(vector<int>& nums, int target) {
    int lft = 0, rht = nums.size() - 1, mid;
    while (lft <= rht) {
        mid = (lft + rht) / 2;
        if (nums[mid] == target) return mid;
        if (is_sorted(nums, lft, mid)) {
            if (target >= nums[lft] && target <= nums[mid]) rht = mid - 1;
            else lft = mid + 1;
        }
        else {
            if (target >= nums[mid] && target <= nums[rht]) lft = mid + 1;
            else rht = mid - 1;
        }
    }
    return -1;
}

int findMin(vector<int>& nums) {
    int lft = 0, rht = nums.size() - 1, mid, rotate_start = nums[0], min;
    while (lft <= rht) {
        mid = (lft + rht) / 2;
        if (rotate_start <= nums[mid]) lft = ++mid;
        else {
            min = nums[mid];
            rht = --mid;
        }
    }
    return min;
}

string decodeString(string s) {
    string res;
    vector<string> vec;
    int i = 0, end = 0;
    string temp;
    while (i < s.size()) {
        temp.clear();
        if (isalpha(s[i])) {
            while (isalpha(s[i])) {
                temp.push_back(s[i]); ++i;
            }
            vec.emplace_back(temp);
        }
        else if (s[i] == '[') {
            vec.emplace_back(string(1, s[i]));
            ++i;
        }
        else if (isdigit(s[i])) {
            while (isdigit(s[i])) {
                temp.push_back(s[i]); ++i;
            }
            vec.emplace_back(temp);
        }
        else {//右括号
            while (1) {
                string ss = vec.back(); vec.pop_back();
                if (ss == "[") {
                    int times = stoi(vec.back());vec.pop_back();
                    string t;
                    for (int i = 0;i < times;++i) t += temp;
                    vec.emplace_back(t);break;
                }
                else temp.insert(0, ss);
            }
            ++i;
        }
    }
    for (int i = 1;i < vec.size();++i)vec[0] += vec[i];
    return vec[0];
}

vector<int> dailyTemperatures(vector<int>& temperatures) {
    int n = temperatures.size();
    vector<int> res(n);
    vector<unsigned> temper(71, 0x7ffffff);//对i索引，表示30+i度温度的最小
    for (int i = n - 1;i >= 0;--i) {
        unsigned warmer = 0x7fffffff;
        for (int t = temperatures[i] + 1; t <= 100; ++t) warmer = min(warmer, temper[t - 30]);
        if (warmer != 0x7fffffff) res[i] = warmer - i;
        temper[temperatures[i] - 30] = i;
    }
    return res;
}


void maxheap(vector<int>& heap, int parent, const int& heapsize) {
    int lft = (parent << 1) + 1, rht = (parent << 1) + 2;
    int max_one = (lft < heapsize && heap[lft] > heap[parent]) ? lft : parent;
    max_one = (rht < heapsize && heap[rht] > heap[max_one]) ? rht : max_one;
    if (max_one != parent) {
        swap(heap[parent], heap[max_one]);
        maxheap(heap, max_one, heapsize);
    }
}

void build_maxheap(vector<int>& heap, const int& heapsize) {
    for (int i = heapsize / 2 - 1; i >= 0; --i) {
        maxheap(heap, i, heapsize);
    }
}

int findKthLargest(vector<int>& nums, int k) {
    int heapsize = nums.size();
    build_maxheap(nums, heapsize);
    for (int i = nums.size() - 1; i >= nums.size() - k + 1; --i) {
        swap(nums[0], nums[i]);
        --heapsize;
        maxheap(nums, 0, heapsize);
    }
    return nums[0];
}

int quickselect(vector<int>& nums, int l, int r, int k) {
    if (l == r) return nums[k];
    int partition = nums[l], i = l - 1, j = r + 1, n = nums.size();
    while (i < j) {
        ++i;--j;
        while (i < n && nums[i] < partition) ++i;
        while (j >= 0 && nums[j] > partition) --j;
        if (i < j) swap(nums[i], nums[j]);
    }
    if (k <= j)return quickselect(nums, l, j, k);//此时j对应小于主元的最后一个元素
    else return quickselect(nums, j + 1, r, k);
}

int findKthLargest_qsort(vector<int>& nums, int k) {
    int n = nums.size();
    return quickselect(nums, 0, n - 1, n - k);
}

void sortColors(vector<int>& nums) {
    int n = nums.size();
    int p0 = -1, p1 = -1;
    for (int i = 0; i < n; ++i) {
        if (nums[i] == 1) {
            swap(nums[i], nums[++p1]);
        }
        else if (nums[i] == 0) {
            swap(nums[i], nums[1 + p0]);
            if (p0 < p1) {
                swap(nums[i], nums[p1 + 1]);
            }
            ++p1;++p0;
        }
    }
}

int tribonacci(int n) {
    if (n <= 1) return n;
    else if (n == 2) return n;
    vector<int> mem = { 0,1,1 ,0 };
    int i = 3;
    while (i <= n) {
        mem[0] = mem[1] + mem[2] + mem[3];
        mem[3] = mem[2];
        mem[2] = mem[1];
        mem[1] = mem[0];
        ++i;
    }
    return mem[0];
}

int minCostClimbingStairs(vector<int>& cost) {
    int n = cost.size();
    vector<int> mem = { 0,0,0 };
    for (int i = 2; i <= n; i++) {
        mem[0] = min(mem[1] + cost[i - 1], mem[2] + cost[i - 2]);
        mem[2] = mem[1];
        mem[1] = mem[0];
    }
    return mem[0];
}

int deleteAndEarn(vector<int>& nums) {
    if (nums.size() == 1) return nums[0];
    sort(nums.begin(), nums.end(), less<int>());
    int n = nums.size(), i = 2;
    unordered_map<int, int> um;
    vector<int> keys = { nums[0] };
    for (auto i : nums) {
        if (!um.count(i)) um[i] = 1;
        else ++um[i];
        if (i != keys.back()) keys.emplace_back(i);
    }
    if (keys.size() == 1) return keys[0] * um[keys[0]];

    int temp;//偷到第二家
    temp = ((keys[1] - 1) != keys[0]) ? keys[0] * um[keys[0]] + keys[1] * um[keys[1]] : max(keys[1] * um[keys[1]], keys[0] * um[keys[0]]);
    vector<int> mem = { temp,temp,keys[0] * um[keys[0]] };//{结果，偷到第二家，偷到第一家}
    while (i < keys.size()) {//进入时已经偷了2家或1家
        if ((keys[i] - 1) != keys[i - 1]) mem[0] = max(mem[1], mem[2]) + keys[i] * um[keys[i]];
        else mem[0] = max(mem[1], mem[2] + keys[i] * um[keys[i]]);
        mem[2] = mem[1];
        mem[1] = mem[0];
        ++i;
    }
    return mem[0];
}

void find_symmetry_helper(const string& s, vector<vector<uint8_t>>& sym_word, int i, int j, int& m, int& st) {
    if (i >= j || sym_word[i][j] == 2) return;
    if (sym_word[i][j] == 1) { m = max(j - i + 1, m);if (m == j - i + 1) st = i; return; }
    else {
        if (j - i == 1) sym_word[i][j] = s[i] == s[j];
        else {
            find_symmetry_helper(s, sym_word, i + 1, j - 1, m, st);
            sym_word[i][j] = (s[i] == s[j] && sym_word[i + 1][j - 1] == 1) ? 1 : 2;
        }
        if (sym_word[i][j] == 1) { m = max(j - i + 1, m);if (m == j - i + 1) st = i; }
    }
}

string longestPalindrome(string s) {
    int n = s.size(), max_len = 1, max_start = 0;
    if (n < 2) return s;
    vector<vector<uint8_t>> dp(n, vector<uint8_t>(n, 0));
    for (int i = 0; i < n; i++) dp[i][i] = 1;
    for (int i = 0; i < n;++i) {
        for (int j = n - 1;j > i;--j) {
            if (j - i + 1 > max_len) find_symmetry_helper(s, dp, i, j, max_len, max_start);
        }
    }
    return s.substr(max_start, max_len);
}

void symmetry_str_helper(const string& s, vector<vector<uint8_t>>& sym_word, int i, int j, int& m) {
    if (i >= j || sym_word[i][j] != 0) return;
    else {
        if (j - i == 1) sym_word[i][j] = s[i] == s[j];
        else {
            symmetry_str_helper(s, sym_word, i + 1, j - 1, m);
            sym_word[i][j] = (s[i] == s[j] && sym_word[i + 1][j - 1] == 1) ? 1 : 2;
        }
        if (sym_word[i][j] == 1) { m = max(j - i + 1, m); }
    }
}

int longestPalindromeSubseq(string s) {
    int len = 0, n = s.size();
    if (n < 2) return n;
    vector<vector<uint8_t>> dp(n, vector<uint8_t>(n, 0));
    for (int i = 0; i < n; i++) dp[i][i] = 1;
    for (int i = 0; i < n;++i) {
        for (int j = n - 1;j > i;--j) {
            if (j - i + 1 > len) symmetry_str_helper(s, dp, i, j, len);
        }
    }
    return len;
}


int longestArithSeqLength2(vector<int>& nums) {
    int n = nums.size();
    int res = 1;
    vector<unordered_map<int, int>> dp(n);
    for (int i = -500; i <= 500; i++) dp[0][i] = 1;//头元素可以视为任意差的数列开头
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            int d = nums[i] - nums[j];
            if (dp[j].count(d)) dp[i][d] = dp[j][d] + 1;
            else dp[i][d] = 2;
            res = max(res, dp[i][d]);
        }
    }
    return res;
}


int minDistance(string word1, string word2) {
    int n = word1.length();
    int m = word2.length();
    if (n * m == 0) return n + m;
    vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));
    for (int i = 0; i < n + 1; i++) dp[i][0] = i;
    for (int j = 0; j < m + 1; j++) dp[0][j] = j;
    for (int i = 1; i < n + 1; i++) {
        for (int j = 1; j < m + 1; j++) {
            int left = dp[i - 1][j] + 1;
            int down = dp[i][j - 1] + 1;
            int left_down = dp[i - 1][j - 1];
            if (word1[i - 1] != word2[j - 1]) left_down += 1;
            dp[i][j] = min(left, min(down, left_down));
        }
    }
    return dp[n][m];
}


int longestArithSeqLength(vector<int>& nums) {
    auto minmax = minmax_element(nums.begin(), nums.end());
    int diff = *minmax.second - *minmax.first;
    auto minit = minmax.first, maxit = minmax.second;
    int res = 1;
    for (int d = -diff; d <= diff; ++d) {
        vector<int> f(*maxit + 1, -1);
        for (int num : nums) {
            if (int prev = num - d; prev >= *minit && prev <= *maxit && f[prev] != -1) {
                f[num] = max(f[num], f[prev] + 1);
                res = max(res, f[num]);
            }
            f[num] = max(f[num], 1);
        }
    }
    return res;
}

int jump(vector<int>& nums) {
    int maxPos = 0, n = nums.size(), end = 0, step = 0, i = 0, max_index = 0;
    while (1) {
        maxPos = i + nums[i];
        if (maxPos >= n - 1) return ++step;
        for (int j = i + 1;j <= maxPos;++j) {
            if (j + nums[j] > max_index + nums[max_index]) {
                max_index = j;
            }
        }
        i = max_index;maxPos = max_index + nums[max_index];
        step++;
    }
    return step;
}


int main() {
    //a_plus_b();
    //cout << longestPalindromeSubseq("xaabacxcabaaxcabaax");
    vector <int> test = { 1,4,8,12,32,44,213,543,654,765,876,987,998 };
    //cout << jump(test);
    vector<vector<char>> grid = { {'1','1','1','1','0'},{'1','1','0','1','0'},{'1','1','0','0','0'},{'0','0','0','0','0'} };
    vector <int> test2 = { 1,3,4 };
    vector<vector<int>> test3 = { {0,1} };
    vector<string> strs = { "eat","tea","tan","ate","nat","bat" };
    ListNode temp(65535);
    ListNode temp2(65535);
    TreeNode* tree1 = new TreeNode;
    ListNode* ln = new ListNode;
    ln = build_nodelist_withouthead(ln, test);
    tree1 = build_tree_from_array(tree1, test, 0);
    //reverseList(build_nodelist_withouthead(&temp, test));
    //isPalindrome(build_nodelist_withouthead(&temp, test));
    //mergeTwoLists(build_nodelist_withouthead(&temp, test), build_nodelist_withouthead(&temp2, test2));
    //inorderTraversal(build_tree_from_array(tree1, test, 1));
    //cout << maxDepth(build_tree_from_array(tree1, test));
    //invertTree2(tree1);
    //cout << isSymmetric(tree1);
    //cout << diameterOfBinaryTree(tree1);
    //auto bst1 = sortedArrayToBST(test);
    //cout << isValid("(){}}{");
    //cout << maxProfit(test);
    //cout << climbStairs(5);
    //groupAnagrams(strs);
    //cout << longestConsecutive(test);
    //threeSum(test);
    //cout << lengthOfLongestSubstring("abcabcbb");
    //prit_container(findAnagrams("cbaebabacd", "abc"));
    //prit_container(findAnagrams("ababababab", "aab"));
    //cout << subarraySum(test, 2);
    //cout << maxSubArray(test);
    //merge_intervals(test3);
    //right_rotate_vector(test, 3);
    //prit_container(productExceptSelf(test));
    //print_linknode(removeNthFromEnd(ln, 2));
    //copyRandomList2(build_random_node(test3));
    //cout << isValidBST(tree1);
    //cout << kthSmallest(tree1, 4);
    //prit_container(rightSideView(tree1));
    //flatten(tree1);
    //cout << pathSum(tree1, 8);
    //permute(test);
    //cout << rob(test);
    //prit_container(spiralOrder(test3));
    //cout << numIslands(grid);
    //cout << canFinish(2, test3);
    //for (auto v : combinationSum(test, 7)) prit_container(v);
    //partition("abbab");
    //prit_container(searchRange(test, -1));
    //cout << search_rotate(test, 1);
    //cout << findMin(test);
    //cout << decodeString("100[leetcode]");
    //prit_container(dailyTemperatures(test));
    //cout << findKthLargest_qsort(test, 4);
    //sortColors(test); prit_container(test);
    //cout << tribonacci(4);
    //cout << deleteAndEarn(test);
    auto a = binarySearchv2(test, 0);
    cout << a.first << ',' << a.second << endl;

    //strings_withdot();
    //auto si = twoSum2(test, 6);
    //moveZeroes(test);
}