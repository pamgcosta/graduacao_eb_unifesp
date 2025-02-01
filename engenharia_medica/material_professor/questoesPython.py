"""
Respostas de questões sobre Python para a Unidade Curricular "Engenharia Médica Aplicada"
Instituto de Ciência e Tecnologia
Universidade Federal de São Paulo
@author: Adenauer G. CASALI (casali@unifesp.br)
"""

## Este código responde algumas das principais questões envolvendo
#métodos e bibliotecas em Python para a solução das atividades 
#práticas do curso. Será atualizado na medida em que questões forem
#surgindo. 


#%% Bibliotecas necessárias:
import numpy as np
import matplotlib.pyplot as plt

#ÍNDICE:
#Q1 - indexar arrays em python (linha 24)
#Q2 - separar classes juntas com o reshape (linha 88)
#Q3 - juntar e separar classes com número variável de classes e características (linha 134)


#%% Q1 - Questão sobre indexação de arrays em python:

#Em aprendizado de máquina é muito comum trabalharmos com
#uma matriz de dados no formato "Características x Padrões" 
# e com um vetor de classes para cada um dos padrões contidos nos dados.
#Por exemplo, suponha que esta seja a matriz de dados
#no formato "Característica x Padrões"
Dados = np.random.rand(5,20)
#E suponha que este seja o vetor com as classes:
classes = np.random.permutation(np.hstack((np.ones([13]), np.zeros([7]))))
#no qual temos 7 padrões com classes =0 e 13 com classes =1. 

#Note que o array com as classes tem um formato de vetor (unidimensional): 
print(np.shape(classes))
#Muitas vezes, porém, o vetor de classes tem uma dimensão do tipo 1 x 20
#ou 20 x 1. Nestes casos, a dimensão "1" será inútil (vazia) e 
# poderá gerar problemas se você não souber lidar com ela. 
# Se esse for o caso, você pode eliminar esta dimensão extra com o 
# commando "squeeze" assim.
classes=np.squeeze(classes)

#Muito bem, considerando o array unidimensional "classes", suponha
# que você queira selecionar apenas os padrões que pertencem
#à classe =0. Você não precisa usar loops para isso! 
# Veja como fazer abaixo:
#O seguinte comando
classes==0
#retorna um array de verdadeiros ou falsos, onde os verdadeiros
# estão localizados nas posições correspondentes aos padrões em 
# que a classe é igual à 0.
#Para extrair estes padrões, você pode usar este vetor lógico como
#um indexador da matriz:
Dados_classe0 = Dados[:,classes==0]
#Note que estamos usando classes==0 na segunda dimensão da matriz
# Dados, pois é nesta dimensão que estão os padrões. 
# Veja que a dimensão da matriz resultante,
np.shape(Dados_classe0)
#é de características por padrões, mas apenas com os padrões 
# dos quais a classe é zero!
#Se você quiser selecionar os padrões com classe 1 é só fazer o
# mesmo procedimento, substituindo 0 por 1:
Dados_classe1 = Dados[:,classes==1]
#Fácil, não?

#Agora suponha que você quer juntar os padrões novamente. 
#Há várias formas de se fazer isso, mas caso você queira preservar 
#as posições das classes (tal que o vetor "classes" continue
# válido), você pode fazer assim: primeiro crie uma matriz com zeros
#na dimensão correta:
Dados_juntos=np.zeros_like(Dados)
#E agora preencha a matriz com os dados de cada classe usando o
#teste lógico como indexador:
Dados_juntos[:,classes==0]=Dados_classe0
Dados_juntos[:,classes==1]=Dados_classe1

#Esta nova matriz corresponde à mesma matriz com os dados originais?
print(np.array_equal(Dados_juntos,Dados))
#Sim! É assim que podemos facilmente separar e juntar classes de 
# padrões.





# %%Q2 - Questão sobre mudanças das dimensões em arrays:

#Por vezes queremos transformar uma matriz bidimensional em
# um vetor. Por exemplo, imagine que você tem uma matriz Nc x Np que 
# corresponde a uma determinada características calculada em um certo 
# número Np de padrões e um certo número Nc de classes, como esta:
CaracteristicaUnica=np.concatenate((np.random.rand(1,20), np.random.rand(1,20)+2))
#Verifique as dimensões:
(Nc,Np)=np.shape(CaracteristicaUnica)
print('A matriz tem '+str(Np)+' padrões em '+str(Nc)+' classes')

#Vamos fazer um gráfico para visualizar as classes:
plt.figure()
plt.plot(CaracteristicaUnica[0,:],'ob')
plt.plot(CaracteristicaUnica[1,:],'or')
plt.xlabel('Número do padrão')
plt.ylabel('Valor da característica')
plt.title('Classes originais')
plt.show()

#Agora suponha que você queira juntar as classes em um vetor unidimensional
#mas sem perder informação de onde elas estão. 
#Para isso, você pode usar o comando reshape:
CaracteristicaUnica_classesjuntas=CaracteristicaUnica.reshape(Nc*Np,)
#Note que o input do reshape é a dimensão dos dados que será neste
#exemplo de um elemento unidimensional (40,) com todos os 40 padroes 
#juntos.
print(np.shape(CaracteristicaUnica_classesjuntas))

#Agora como separar as classes de volta? Use o reshape (veja a 
# diferença nos itens entre parênteses, agora a matriz será 2 x 20):
CaracteristicaUnica_classesseparadas=CaracteristicaUnica_classesjuntas.reshape(Nc,Np)
print(np.shape(CaracteristicaUnica_classesseparadas))

#Será que as classes foram misturadas? Vamos conferir fazendo o 
#mesmo gráfico de antes:
plt.figure()
plt.plot(CaracteristicaUnica_classesseparadas[0,:],'ob')
plt.plot(CaracteristicaUnica_classesseparadas[1,:],'or')
plt.xlabel('Número do padrão')
plt.ylabel('Valor da característica')
plt.title('Depois de Juntar e Separar')
plt.show()



#%% Q3 - Questão sobre juntar e separar múltiplas classes em matrizes.

#A melhor forma de trabalhar com dados em Aprendizado de Máquina
#Supervisionado é usar uma matriz contendo todos os padrões
#de todas as características e um vetor com a informação das classes.
#Mas por vezes partimos de classes separadas e precisamos juntar e separá-las
# com flexibilidade. 
#Para ver como fazer isso, siga os passos abaixo,

#Suponha que tenhamos 3 classes distintas com 5 características e 90 padrões no total:
classe0 = np.random.rand(5,20)
classe1 = np.random.rand(5,30)
classe2 = np.random.rand(5,40)
 

#Vamos juntar as três classes. Para isso, vamos definir uma função que
# poderá ser usada em outros problemas do curso se necessário:

def juntar(dados):
    #Nesta função, a variável "dados" deve ser uma lista de classes. 
    # Cada elemento da lista corresponde
    #a um array com padrões x características (formato N x L), onde L é fixo
    #para todas as classes.
    #A função retorna uma matriz padrões x características, contendo 
    # todos os padrões de todas as classes e um vetor contendo o número da 
    # classe de cada padrão.

    Nc=len(dados) #número de classes
    #verificando se as dimensões estão corretas:
    L=np.unique([np.size(s,axis=1) for s in dados]) #número de características
    if len(L)>1:
        print('ERRO: todas classes devem ter o mesmo número de características!')

    dados_todos=dados[0]
    classes_todos=np.zeros(np.size(dados[0],axis=0),)
    for i in range(1,Nc):
        dados_todos=np.concatenate((dados_todos,dados[i]))
        classes_todos=np.concatenate((classes_todos,i+np.zeros(np.size(dados[i],axis=0),)))
    return dados_todos,classes_todos

#Para usar esta função, vamos colocar cada classe dentro de uma lista. 
#Note que a função exige dados no formao padrões x características, mas nossas 
# classes estão no formato características x padrões. Então 
#precisamos também transpor as matrizes (método "T"):
dados=[]
dados.append(classe0.T)
dados.append(classe1.T)
dados.append(classe2.T)

#Agora vamos chamar a função para juntar as classes:
dados_todos,classes_todos=juntar(dados)

#Veja a dimensão do output com os dados:
print(np.shape(dados_todos))
#Ou seja, temos os padrões de todas as classes juntos na primeira dimensão.

#Note que as classes estão no vetor "classes_todos":
print(np.shape(classes_todos))

#Com as classes juntas podemos, por exemplo, normalizar as características ou
#realizar outros procedimentos. 

#E se for necessário separá-las novamente, basta usar a técnica vista na Q1 acima:
dados_classe0=dados_todos[classes_todos==0,:]
dados_classe1=dados_todos[classes_todos==1,:]
dados_classe2=dados_todos[classes_todos==2,:]

#Veja como, após transpormos as matrizes, obtemos os dados separados corretamente, 
# exatamente como as classes originais:
print(np.array_equal(dados_classe0.T,classe0))
print(np.array_equal(dados_classe1.T,classe1))
print(np.array_equal(dados_classe2.T,classe2))


