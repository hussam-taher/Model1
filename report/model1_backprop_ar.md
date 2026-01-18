# Model 1: Backpropagation (Arabic draft)

Model 1: Backpropagation
(Start with Code — based on free sources)
✅ الكود المستخدم (ملفات جاهزة)
•	backprop_numpy_wdbc.py (تدريب MLP بــ NumPy باستخدام Backprop)
•	plot_results.py (توليد الرسوم Learning Curves + Confusion Matrix)
•	نواتج محفوظة:
o	results.json (كل الأرقام والـ hyperparameters)
o	history.json (loss/accuracy عبر epochs)
o	learning_curves_loss.png
o	learning_curves_accuracy.png
o	confusion_matrix.png
نتائج التشغيل الفعلية (Test):
•	Accuracy = 0.9651
•	Precision = 0.9811
•	Recall = 0.9630
•	F1-score = 0.9720
•	Confusion Matrix = [[31, 1], [2, 52]]
مصدر بيانات WDBC مجاني من UCI + موثّق في scikit-learn. [1], [2]
طريقة backprop نفسها مرجعية من مصادر مجانية (Rumelhart PDF مجاني + كتاب DeepLearningBook مجاني + Nielsen مجاني). [6], [7], [8]
________________________________________
2. Dataset Description (Backpropagation)
• Source of the dataset:
Breast Cancer Wisconsin (Diagnostic) (WDBC) من UCI Machine Learning Repository (مجاني). [1]
كما تم استخدام النسخة الجاهزة عبر sklearn.datasets.load_breast_cancer والمذكور فيها أنها نسخة من UCI. [2]
• Type of data used:
بيانات عددية (Numerical tabular data) (Features real-valued). [1], [2]
• Dataset size and features:
•	عدد العينات: 569
•	عدد الخصائص: 30
•	المهمة: تصنيف ثنائي (malignant/benign). [1], [2]
• Data preprocessing steps:
•	Data cleaning: لا توجد قيم مفقودة في النسخة المستخدمة من scikit-learn (البيانات تأتي جاهزة). [2]
•	Normalization or scaling: استخدام StandardScaler (fit على train ثم transform على val/test). [4]
•	Tokenization: غير مطبق (ليس نص).
•	Train-validation-test split: تقسيم 70% تدريب، ثم 30% تُقسّم بالتساوي إلى Validation و Test (مع stratify لضمان توازن الفئات). [3]
• Justification for choosing the dataset:
البيانات مناسبة لنموذج تعليمي يوضّح Backpropagation لأن المهمة تصنيف ثنائي واضح مع عدد خصائص محدود، مما يسمح بإظهار أثر التدريب وقياس الأداء بمقاييس متعددة بسرعة. [1], [2]
________________________________________
3. Implementation Environment (Backpropagation)
• Programming language used:
Python. [9]
• Frameworks and libraries:
•	NumPy (حسابات مصفوفية). [10]
•	scikit-learn (تحميل البيانات + splitting + StandardScaler + metrics). [2], [3], [4], [5]
•	Matplotlib (الرسوم والمنحنيات). [11]
• Platform where the code was executed:
Local machine (حسب قراركم). (والتوثيق يظل صحيح لأن الأدوات كلها تعمل لوكل). [9], [12]
• Hardware details (CPU/GPU/TPU):
CPU-only كافٍ لهذا النموذج لأن البيانات صغيرة والشبكة بسيطة. [7]
________________________________________
4. Model Descriptions and Code Analysis (Backpropagation)
4.1 Theoretical Background
• Brief explanation of how the model works:
Backpropagation يحسب gradients لدالة الخسارة بالنسبة للأوزان بكفاءة باستخدام Chain Rule ثم يتم تحديث الأوزان لتقليل الخسارة. [6], [7], [8]
• Core components of the architecture:
في هذا التطبيق تم استخدام شبكة بسيطة (MLP):
•	طبقة إدخال (30 feature)
•	طبقة مخفية (ReLU)
•	طبقة إخراج (Sigmoid) لتصنيف ثنائي. [7], [8]
• Strengths and limitations:
•	القوة: يسمح بتدريب شبكات متعددة الطبقات عمليًا ويُعد الأساس لتدريب الشبكات العصبية الحديثة. [6], [7]
•	القيود: قد يواجه vanishing gradients في بعض الإعدادات (خاصة مع تفعيلات معينة أو عمق كبير). [7], [8]
• Typical applications:
يستخدم كأساس تدريب لمعظم الشبكات العصبية (Feedforward/CNN/RNN/Transformers). [7]
4.2 Code Implementation
• Explanation of the model architecture used in the code:
MLP بطبقة مخفية واحدة + ReLU + إخراج Sigmoid. [7], [8]
• Description of layers, activation functions, loss functions, and optimizers:
•	Hidden activation: ReLU (مع مشتقها في backprop). [7]
•	Output activation: Sigmoid. [7], [8]
•	Loss: Binary Cross-Entropy. [7]
•	Optimizer: Gradient Descent (تحديث مباشر للأوزان باستخدام lr). [6], [7]
• Hyperparameters used:
حسب results.json:
•	hidden_units = 32
•	learning_rate = 0.05
•	epochs = 300
•	random_state = 42
• Explanation of the training process:
Forward pass → حساب الخسارة → backward pass لحساب gradients → تحديث الأوزان → تكرار عبر epochs مع تسجيل train/val loss & accuracy. [6], [7], [8]
4.3 Training and Execution
• How the code was executed step-by-step:
1.	تحميل dataset من scikit-learn [2]
2.	split باستخدام train_test_split + stratify [3]
3.	scaling باستخدام StandardScaler [4]
4.	تدريب الشبكة بــ backprop (NumPy) [6], [8]
5.	تقييم test metrics باستخدام scikit-learn metrics [5]
• Training time and computational cost:
زمن التدريب الفعلي المسجل في results.json: ≈ 17.93 ثانية (CPU). (يعتمد على جهازك)
• Challenges faced:
•	في بداية التجربة، قلة epochs مع Sigmoid في الطبقة المخفية أدت إلى أداء ضعيف (bias نحو فئة واحدة). تم تحسين ذلك بتغيير activation للطبقة المخفية إلى ReLU وزيادة epochs وتحسين lr. [7]
________________________________________
5. Results and Performance Evaluation (Backpropagation)
• Training and validation accuracy/loss:
من results.json/history (آخر Epoch = 300):
•	Train loss ≈ 0.0768, Train acc ≈ 0.9824
•	Val loss ≈ 0.1998, Val acc ≈ 0.9647
• Test performance:
•	Accuracy = 0.9651
•	Precision = 0.9811
•	Recall = 0.9630
•	F1 = 0.9720
•	Confusion Matrix = [[31, 1], [2, 52]] [5]
• Graphs and visualizations of learning curves:
تم توليد الرسوم وحفظها:
•	learning_curves_loss.png
•	learning_curves_accuracy.png
•	confusion_matrix.png
(باستخدام Matplotlib). [11]
• Evaluation metrics used:
Accuracy / Precision / Recall / F1 + Confusion Matrix عبر scikit-learn. [5]
• Interpretation of the results:
القيم تشير إلى أداء قوي على الاختبار، مع أخطاء قليلة (FP=1, FN=2)، ويظهر من منحنيات التعلم أن loss ينخفض مع تحسن دقة التحقق، مما يدل على تعلّم مستقر في هذا الإعداد. [5]
________________________________________
Optional Enhancements (Bonus) ✅ (توضع بعد الفقرة 5)
هذه إضافات اختيارية للحصول على درجات إضافية:
1.	Use of additional evaluation metrics:
تمت إضافة Precision/Recall/F1 و Confusion Matrix. [5]
2.	Experimenting with different hyperparameters:
تمت تجربة إعدادات (epochs/activation/learning rate) وتحسين الأداء مقارنة بإعدادات أولية ضعيفة. [7]
3.	Adding ablation studies or error analysis:
(جاهز للكتابة) تحليل أخطاء FP/FN من Confusion Matrix وتفسير سببها. [5]
4.	Including a GitHub repository link for the code:
(سنضيف الرابط بعد رفع الريبو) — وهذا ضمن متطلبات البونص.
5.	Fine-tuning BERT:
هذا خاص بنموذج BERT وسننفذه في Model 5 فقط. [13]
