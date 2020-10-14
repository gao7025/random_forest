
# random_forest
Use random forest to build machine learning model, and use grid search for optimization

## 机器学习-随机森林的网格调参实例

=====================================================================
### **1. 随机森林**

RandomForestClassifier官方网址：
*<https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>*

#### **1.1 原理解释**

从给定的训练集通过多次随机的可重复的采样得到多个 bootstrap 数据集。接着，对每个 bootstrap 数据集构造一棵决策树，构造是通过迭代的将数据点分到左右两个子集中实现的，这个分割过程是一个搜索分割函数的参数空间以寻求最大信息增量意义下最佳参数的过程。然后，在每个叶节点处通过统计训练集中达到此叶节点的分类标签的直方图经验的估计此叶节点上的类分布。这样的迭代训练过程一直执行到用户设定的最大树深度（随机森林提出者Breiman采用的是ntree=500）或者直到不能通过继续分割获取更大的信息增益为止。

简而言之：

随机森林在bagging基础上做了修改，从样本集中用bootstrap采样选出n个样本，从所有属性中随机选择k个属性，选择最佳分割属性作为节点建立CART决策树，重复以上两步m次，即建立了m棵CART决策树，这m个CART形成了随机森林，通过投票表决结果，决定数据属于哪一类。

*class sklearn.ensemble.RandomForestClassifier(n_estimators=100, , criterion=‘gini’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=‘auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)[source]*

#### **1.2 Parameters**

(1)n_estimators：int, default=100
森林中树的个数。这个属性是典型的模型表现与模型效率成反比的影响因子，即便如此，你还是应该尽可能提高这个数字，以让你的模型更准确更稳定。

(2)criterion：{“gini”, “entropy”}, default=”gini”
度量分裂的标准。可选值：“mse”，均方差（mean squared error）；“mae”，平均绝对值误差（mean absolute error）
支持的标准是基尼杂质的“gini（基尼）”和信息增益的“entropy（熵）”。注意：此参数是特定于树的。

(3)max_depth：int, default=None
integer或者None。树的最大深度，如果None，节点扩展直到所有叶子是纯的或者所有叶子节点包含的样例数小于min_samples_split

(4)min_samples_split：int or float, default=2
分裂内部节点需要的最少样例数。int(具体数目),float(数目的百分比)

(5)min_samples_leaf：int or float, default=1
叶子节点上应有的最少样例数。int(具体数目),float(数目的百分比)。
更少的节点数使得模型更容易遭受noise data的影响，我通常设置这个值大于50，但是你需要寻找最适合你的数值。

(6)min_weight_fraction_leaf：float, default=0.0
The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.

(7)max_features：{“auto”, “sqrt”, “log2”}, int or float, default=”auto”
寻找最佳分裂点时考虑的特征数目。可选值，int（具体的数目），float（数目的百分比），string（“auto”， “sqrt”，“log2”）.
这一属性是对单个树来设置的，通常来讲，这个值越大单棵树可以考虑的属性越多，则模型的表现就越好。但是这也不是肯定的，不过有一点是肯定的，增加这个值会导致算法运行速度变慢，所以需要我们考虑去达到一个平衡。

(8)max_leaf_nodes：int, default=None
以”最优优先方式”(best-first fashion),最优节点定义为:纯度的相对减少.如果None则不限制叶子节点个数

(9)min_impurity_decrease：float, default=0.0
一个阈值,表示一个节点分裂的条件是:如果这次分裂纯度的减少大于等于这这个值.

(10)min_impurity_split：float, default=None
树增长提前结束的阈值.对于当前节点,大于这个阈值将分裂,否则就看做叶子节点

(11)bootstrap：bool, default=True
构建数是不是采用有放回样本的方式(bootstrap samples)

(12)oob_score：bool, default=False
oob_score ：交叉验证相关的属性。

(13)n_jobs：int, default=None
设定fit和predict阶段并列执行的任务个数,如果设置为-1表示并行执行的任务数等于计算级核数; [integer, optional (default=1)]

(14)random_state：int or RandomState, default=None
如果是int数值表示它就是随机数产生器的种子.如果指定RandomState实例,它就是随机产生器的种子.如果是None,随机数产生器是
np.random所用的RandomState实例; [int, RandomState instance or None, optional (default=None)]

(15)verbose：int, default=0
控制构建数过程的冗长度; [int, optional (default=0)]

(16)warm_start：bool, default=False
当设置为True,重新使用之前的结构去拟合样例并且加入更多的估计器(estimators,在这里就是随机树)到组合器中;

(17)class_weight：{“balanced”, “balanced_subsample”}, dict or list of dicts, default=None
“banlanced”模式是根据y标签值自动调整权值与输入数据的类频率成反比,计算公式是:n_samples / (n_classes np.bincount(y)).“balanced_subsample”模式的与”balanced模式相同,只不过在每一次树增长过程中权值的计算是根据有放回样本的.

(18)ccp_alphanon-negative：float, default=0.0
用于最小化成本复杂性修剪的复杂性参数。 将选择成本复杂度最大且小于ccp_alpha的子树。 默认情况下，不执行修剪。 有关详细信息，请参见最小化成本复杂性修剪。

(19)max_samples：int or float, default=None
如果bootstrap为True，则从X抽取以训练每个基本估计量的样本数。

#### **1.3 Attributes**

(1)base_estimator_：DecisionTreeClassifier
子分类器用于创建拟合子分类器的集合。

(2)estimators_：list of DecisionTreeClassifier
拟合子分类器的集合

(3)classes_：ndarray of shape (n_classes,) or a list of such arrays
类标签（单输出问题）或类标签数组的列表（多输出问题）

(4)n_classes_：int or list
类数（单个输出问题），或包含每个输出的类数的列表（多输出问题）

(5)n_features_：int
执行拟合时的特征数量

(6)n_outputs_：int
执行拟合时的输出数量

(7)feature_importances_：ndarray of shape (n_features,)
基于impurity的特征的重要性

(8)oob_score_：float
使用袋外估计获得的训练数据集的分数。 仅当oob_score为True时，此属性才存在

(9)oob_decision_function_：ndarray of shape (n_samples, n_classes)
使用训练集上的实际估计值计算的决策函数。 如果n_estimators小，则有可能在引导过程中从未遗漏任何数据点。 在这种情况下，oob_decision_function_可能包含NaN。 仅当oob_score为True时，此属性才存在。

### **2. 网格搜索**

GridSearchCV官方网址：
*<https://scikit-*learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV>**

#### **2.1 原理解释**

GridSearchCV，它存在的意义就是自动调参，只要把参数输进去，就能给出最优化的结果和参数。但是这个方法适合于小数据集，一旦数据的量级上去了，很难得出结果。这个时候就是需要动脑筋了。数据量比较大的时候可以使用一个快速调优的方法——坐标下降。它其实是一种贪心算法：拿当前对模型影响最大的参数调优，直到最优化；再拿下一个影响最大的参数调优，如此下去，直到所有的参数调整完毕。这个方法的缺点就是可能会调到局部最优而不是全局最优，但是省时间省力，巨大的优势面前，还是试一试吧，后续可以再拿bagging再优化。

通常算法不够好，需要调试参数时必不可少。比如SVM的惩罚因子C，核函数kernel，gamma参数等，对于不同的数据使用不同的参数，结果效果可能差1-5个点，sklearn为我们提供专门调试参数的函数grid_search。

*class sklearn.model_selection.GridSearchCV(estimator, param_grid, , scoring=None, n_jobs=None, iid=‘deprecated’, refit=True, cv=None, verbose=0, pre_dispatch='2n_jobs’, error_score=nan, return_train_score=False)[source]*

#### **2.2 Parameters**

(1)estimator：estimator object.
选择使用的分类器，并且传入除需要确定最佳的参数之外的其他参数。每一个分类器都需要一个scoring参数，或者score方法：estimator=RandomForestClassifier(min_samples_split=100,min_samples_leaf=20,max_depth=8,max_features=‘sqrt’,random_state=10),

(2)param_grid：dict or list of dictionaries
需要最优化的参数的取值，值为字典或者列表，例如：param_grid =param_test1，param_test1 = {‘n_estimators’:range(10,71,10)}。

(3)scoring：str, callable, list/tuple or dict, default=None
模型评价标准，默认None,这时需要使用score函数；或者如scoring=‘roc_auc’，根据所选模型不同，评价准则不同。字符串（函数名），或是可调用对象，需要其函数签名形如：scorer(estimator, X, y)；如果是None，则使用estimator的误差估计函数。具体取值可访问*https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter*

(4)n_jobs：int, default=None
n_jobs: 并行数，int：个数,-1：跟CPU核数一致, 1:默认值

(5)pre_dispatch：int, or str, default=n_jobs
指定总共分发的并行任务数。当n_jobs大于1时，数据将在每个运行点进行复制，这可能导致OOM，而设置pre_dispatch参数，则可以预先划分总共的job数量，使数据最多被复制pre_dispatch次

(6)iid：bool, default=False
iid:默认True,为True时，默认为各个样本fold概率分布一致，误差估计为所有样本之和，而非各个fold的平均。

(7)cvint, cross-validation generator or an iterable, default=None
交叉验证参数，默认None，使用三折交叉验证。指定fold数量，默认为3，也可以是yield训练/测试数据的生成器。

(8)refit：bool, str, or callable, default=True
默认为True,程序将会以交叉验证训练集得到的最佳参数，重新对所有可用的训练集与开发集进行，作为最终用于性能评估的最佳模型参数。即在搜索参数结束后，用最佳参数结果再次fit一遍全部数据集。

(9)verbose：integer
verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：偶尔输出，>1：对每个子模型都输出。

(10)error_score：‘raise’ or numeric, default=np.nan
如果估算器拟合出现错误，则分配给分数的值。 如果设置为“ raise”，则会引发错误。 如果给出数值，则引发FitFailedWarning。 此参数不会影响重新安装步骤，这将始终引发错误。

(11)return_train_score：bool, default=False
如果“False”，cv_results_属性将不包括训练分数

#### **2.3 Attributes**

(1)cv_results_ : dict of numpy (masked) ndarrays
具有键作为列标题和值作为列的dict，可以导入到DataFrame中。注意，“params”键用于存储所有参数候选项的参数设置列表。

(2)best_estimator_ : estimator
通过搜索选择的估计器，即在左侧数据上给出最高分数（或指定的最小损失）的估计器。 如果refit = False，则不可用。

(3)best_score_ : float
best_estimator的分数

(4)best_params_ : dict
在保存数据上给出最佳结果的参数设置

(5)best_index_ : int 对应于最佳候选参数设置的索引（cv_results_数组）。
与最佳候选参数设置相对应的（cv_results_数组的）索引。
search.cv_results _ [‘params’] [search.best_index_]上的字典给出了最佳模型的参数设置，该模型给出了最高的平均分数（search.best_score_）。
对于多指标评估，仅当指定重新安装时才存在。

(6)scorer_ : function or a dict
在保留的数据上使用记分器功能，以为模型选择最佳参数。
对于多指标评估，此属性包含将得分者键映射到可调用的得分者的有效得分dict。

(7)n_splits_ : int
交叉验证拆分的数量（折叠/迭代）。

(8)refit_time_：float
用于在整个数据集中重新拟合最佳模型的秒数。仅当改装不为False时才存在。0.20版中的新功能。



