import pandas as pd
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score

y_train=['恶性','恶性','恶性','恶性','恶性','恶性','良性','良性','良性','良性']
y_pre_A=['恶性','恶性','恶性','恶性','恶性','恶性','良性','良性','良性','良性']
y_pre_B=['恶性','恶性','恶性','恶性','恶性','恶性','良性','恶性','恶性','恶性']

cm_A=confusion_matrix(y_train,y_pre_A,labels=['恶性','良性'])
cm_B=confusion_matrix(y_train,y_pre_B,labels=['恶性','良性'])

label=['良性(预测)','恶性(预测)']
index=['良性(实际)','恶性(实际)']


df_A=pd.DataFrame(cm_A,index=index,columns=label)
df_B=pd.DataFrame(cm_B,index=index,columns=label)

print(df_A)
print()
print(df_B)
print()

print(f'模型A 精确率:{precision_score(y_train,y_pre_A,pos_label='恶性')}')
print(f'模型A 召回率:{recall_score(y_train,y_pre_A,pos_label='恶性')}')
print(f'模型A F1值:{f1_score(y_train,y_pre_A,pos_label='恶性')}')
print()
print(f'模型B 精确率:{precision_score(y_train,y_pre_B,pos_label='恶性')}')
print(f'模型B 召回率:{recall_score(y_train,y_pre_B,pos_label='恶性')}')
print(f'模型B F1值:{f1_score(y_train,y_pre_B,pos_label='恶性')}')