
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Biblioteca de códigos em Python para a Unidade Curricular "Engenharia Médica Aplicada"
Instituto de Ciência e Tecnologia
Universidade Federal de São Paulo
@author: Adenauer G. CASALI (casali@unifesp.br)
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import itertools
import scipy.fft as fft
from scipy.signal import welch
from sklearn import svm

def t2_extraicarac(sin,freq,bandas,method='fft',nwelch=5):
  #Extrai características estatísticas e espectrais de um conjunto de sinais de mesma dimensão temporal.
  #Inputs:
  #   - sin = numpy array (num de sinais x tempo)
  #   - freq = frequencia de amostragem dos sinais (Hz)
  #   - bandas = dicionario com a informação das bandas de frequencia a serem extraídas. 
  #              Formato:{'nome da banda (string)':[freqinicial, freqfinal]}
  #              Exemplo: 
  #              bandas={'delta 1':[0.5,2.5],'delta 2':[2.5,4],'teta 1':[4,6],'teta 2':[6,8], 'alfa':[8,12],'beta':[12,20],'gama':[20,45]}
  #   - method = 'fft' or 'welch' (se welch, "nwelch" é o numero de trechos no qual o sinal é dividido)
  #Output: 
  #   - retorna um array de trechos x características e uma lista com os nomes das 
  # caracteristicas correspondentes ao array
    
  (S,X)=np.shape(sin) #S = numero de sinais sinais; X = tamanho dos sinais no tempo
  nc=8+len(bandas) #numero de caracteristicas que serao extraidas
  car=np.zeros((S,nc)) #matriz das caracteristicas
  nomesc=[None]*nc  
  for s in range(S):
    #média
     car[s,0]=np.mean(sin[s,:])
     nomesc[0]='media'

    #variancia
     var0=np.var(sin[s,:],ddof=1)
     car[s,1]=var0
     nomesc[1]='variancia'

    #mobilidade
     x1=np.diff(sin[s,:])
     var1=np.var(x1,ddof=1)
     mob=var1/var0
     car[s,2]=mob
     nomesc[2]='mobilidade'
          
    #complexidade estatística
     x2=np.diff(x1)
     var2=np.var(x2,ddof=1)
     ce=(var2/var1-var1/var0)**(1/2)
     car[s,3]=ce
     nomesc[3]='complexidade'

    ##calculando o espectro:
     if method=='fft':
       yf = np.abs(fft.rfft(sin[s,:]-car[s,0]))**2 
       yf=yf/X
       yf=yf[0:int(np.floor(X/2)+1)]
       xf = np.linspace(0.0, 1.0/(2.0/freq), len(yf))  
     elif method=='welch':
       xf,yf = welch(sin[s,:]-car[s,0],freq,nperseg=X//nwelch)   
     Yf=yf/np.sum(yf) 

    #frequência central do espectro
     car[s,4]=np.sum(xf*Yf)
     nomesc[4]='f-central'

    #potencia na frequencia central
     ifc=np.abs(xf-car[s,4]).argmin()
     car[s,5]=yf[ifc]
     nomesc[5]='P na fc'

    #largura de banda do espectro
     car[s,6]=np.sqrt(np.sum(((xf-car[s,4])**2)*Yf))
     nomesc[6]='l-banda'
    
    #frequência de margem do espectro
     sw=np.cumsum(Yf)
     f=np.max(np.where(sw<=0.9)[0])
     car[s,7]=xf[f]
     nomesc[7]='f-margem'

    #potências espectrais normalizadas nas seguintes bandas: 
    #delta 1 (0.5 a 2.5Hz)
     for ib, b in enumerate(bandas):
        car[s,8+ib]=sum(Yf[((xf>=bandas[b][0]) & (xf<=bandas[b][1]))])
        nomesc[8+ib]='%'+b

  return (car,nomesc)


def t3_remoutliers(padroes,p,method='desvio'):
   #Encontra outlires baseando-se em dois métodos possiveis:
    #  method = 'desvio': mediana +-p x desvio
    #  method = 'quartis': quartis  +-p x intervalo entre quartis
    # padroes = numpy array de uma característica (N x 1), (1 x N), (N,)
    # p = numero de desvios ou de intervalos entre quartis a ser empregado 
    # retorna lista com as posicoes dos outliers no array
    if method =='desvio':
        md=np.median(padroes)
        std=np.std(padroes,ddof=1)
        th1=md+p*std
        th2=md-p*std
    elif method=='quartis':
        q3, q1 = np.percentile(padroes, [75 ,25])
        iqr=q3-q1
        th1=q3+p*iqr
        th2=q1-p*iqr
    outliers=(padroes>th1) | (padroes<th2)
    outs=[i for i, val in enumerate(outliers) if val]
    return outs

def t3_normaliza(dados,metodo='linear',r=1):
    #Realiza a normalizacao de um conjunto de padroes
    # dados = numpy array com padroes de uma caracteristica (N x 1), (1 x N), (N,)
    # metodo ='linear' : normalizacao linear (padrao)
    #        = 'mmx': limitada entre -1 e 1
    #        = 'sfm': rescala nao linear no intervalo 0 a 1
    # r = parametro do metodo sfm (padrao =1)
    #A função retorna os dados normalizados
    if metodo=='linear':
        M=np.mean(dados)
        S=np.std(dados,ddof=1)
        dadosnorm=(dados-M)/S
    elif metodo=='mmx':
        dadosnorm=2*dados/(np.max(dados)-np.min(dados))
        dadosnorm=dadosnorm - (np.min(dadosnorm)+1)
    elif metodo=='sfm':
        x=dados-np.mean(dados)
        x=-x/(r*np.std(dados,ddof=1))
        dadosnorm=1/(1+np.exp(x))        
    return dadosnorm

def t3_preselec(dados1,dados2,alfa,verbose=True):
    #A função retorna os indices das características que apresentaram 
    #significatividade ("rel") e os p-values das características na 
    #distinção entre classes ("p")
    #Inputs:
    # - dados1 = array características x padrões da primeira classe
    # - dados2 = array características x padrões da segunda classe
    # - alfa = taxa de erro tipo I do teste (por exemplo, alfa=0.05)
    #Outputs:
    # - rel = lista com os indices das características relevantes (significativas)
    # - p = valor p do teste estatístico para cada característica
    if len(np.shape(dados1))==1: #controle para caso uma única característica tenha sido carregada sem a dimensão apropriada
        dados1=np.array([dados1])
        dados2=np.array([dados2])
    Ncarac,Npad=dados1.shape
    Ncarac2,Npad2=dados2.shape
    if Ncarac2!= Ncarac:
        print('Erro: matrizes devem ter o mesmo numero de caracteristicas!')
        return
    p=np.zeros(Ncarac)
    for i in range(Ncarac):
        s1=st.shapiro(dados1[i,:])
        s2=st.shapiro(dados2[i,:])
        if (s1[1]<0.05) | (s2[1]<0.05):
            res=st.ranksums(dados1[i,:],dados2[i,:])   
            p[i]=res.pvalue
            if verbose:
                print('Aviso: normalidade rejeitada para a caracteristica nº '+ str(i+1))
        else:
            res=st.ttest_ind(dados1[i,:],dados2[i,:])
            p[i]=res.pvalue
    relevantes=(p<alfa)
    rel=[i for i,val in enumerate(relevantes) if val]
    return rel,p


def t3_fazroc(dados1,dados2,nomecarac='',plotar=True):
    #função que calcula a ROC para ser aplicada em dois conjuntos de padroes (uma caracteristica)
    #Inputs:
    # - dados1 = numpy array com padroes de uma caracteristica para a primeira classe N1 x 1
    # - dados2 = numpy array com padroes de uma caracteristica para a segunda classe N2 x 1
    # - nomecarac = nome da característica para o gráfico
    # - plotar = faz o gráfico da ROC (True or False)
    #Outputs:
    # - auc = área embaixo da curva ROC
    # - fpr = taxas de falsos positivos em função do limiar
    # - tpr = taxas de verdadeiros positivos em função do limiar
    # - acuracia = acuracias totais em função do limiar
    # - thresholds = limiares da roc
    # - classes = classes dos dois inputs (1 ou -1)
    Np1=dados1.shape[0]
    Np2=dados2.shape[0]
    classes=np.zeros(Np1+Np2)
    classes[0:Np1]=-1
    classes[Np1:Np1+Np2]=1
    dados=np.squeeze(np.concatenate((dados1,dados2),axis=0))
    s=dados.argsort()
    thresholds=dados[s]
    classessort=classes[s]
    tpr=np.zeros(len(classes)+1)
    fpr=np.zeros(len(classes)+1)
    acuracia=np.zeros(len(classes)+1)
    erro=np.zeros(len(classes)+1)
    for i in range(len(classessort)):
        pos=classessort[i:]
        tpr[i]=sum(pos==1)/sum(classes==1)
        fpr[i]=sum(pos==-1)/sum(classes==-1)
        acuracia[i]=(sum(pos==1) + (sum(classes==-1)-sum(pos==-1)))/(sum(classes==1)+sum(classes==-1))
        erro[i]=(sum(pos==-1) + (sum(classes==1)-sum(pos==1)))/(sum(classes==1)+sum(classes==-1))
    auc=-np.trapz(tpr,x=fpr)
    classes=[-1,1] #classes dos dois input
    if auc<0.5:
        auc=1-auc
        fpr2=fpr
        fpr=tpr
        tpr=fpr2
        acuracia2=acuracia
        acuracia=erro
        erro=acuracia2
        classes=[1,-1] #classes dos dois inputs
    if plotar:
            plt.figure()
            plt.plot(fpr,tpr)
            plt.title(nomecarac+' AUC = '+'{:.2}'.format(auc))
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.show(block=False)
    return auc, fpr, tpr, classes, thresholds, acuracia

def t3_FDR(dado1,dado2):
    #Calcula o critério FDR entre duas matrizes caracteristicas x padroes
    #Inputs:
    # - dado1 = característica x padrões, classe 1
    # - dado2 = característica x padrões, classe 2
    #Outputs:
    # - fdr = valores de FDR para cada característica
    m1=np.mean(dado1,1)
    m2=np.mean(dado2,1)
    s1=np.var(dado1,axis=1,ddof=1)
    s2=np.var(dado2,axis=1,ddof=1)
    fdr=(m1-m2)**2/(s1+s2)
    return fdr

def t3_selescalar(correl,criterio,peso):
    #Realiza a seleção escalar de características a partir de uma matriz de correlações e um vetor de critérios
    #Inputs:
    # - correl = matriz de correlações entre as características
    # - criterio = vetor de critérios (exemplo: AUC para cada característica, FDR para cada característica)
    # - peso = array com dois elementos [A,B], contendo o peso do critério(A) e o da correlação (B)
    #Outputs:
    # - ordem = ordem das características para seleção 
    Nc=len(criterio)
    Nc1,Nc2=np.shape(correl)    
    if Nc!= Nc1 | Nc!=Nc2:
        print('Confira as dimensoes!')
        return
    falta=list(np.indices(np.shape(criterio))[0])
    ordem=[criterio.argmax().item()] #inicializacao
    falta.remove(ordem)
    while len(falta)>0:  
        x=np.zeros(len(falta))
        for j in range(len(falta)):#pegue uma caracteristica
            x[j]=criterio[falta[j]]*peso[0]
            y=abs(correl[falta[j],ordem])
            x[j]=x[j]-peso[1]*y.mean()
        nordem=falta[x.argmax().item()]
        ordem.append(nordem)
        falta.remove(nordem)
    return ordem

def t3_matrizesdeespalhamento(classes,selcars):
    #Calcula as matrizes de espalhamento a partir de uma lista de classes
    #Inputs:
    # - classes = lista em que cada elemento corresponde a um array características x padrões de cada classe
    # - selcars = índice de quais características usar no cálculo das matrizes
    #Outputs:
    # - matrizes de espalhamento: SW, SM e SB 
    Ncl=len(classes)
    mean_vectors=[]
    for cl in range(Ncl):
        mean_vectors.append(np.mean(classes[cl][selcars,:],axis=1))
    prob=np.zeros(Ncl)
    N=0
    for i in range(Ncl):
        prob[i]=np.size(classes[i][selcars,:],axis=1)
        N=N+prob[i]
    prob=prob/N
    Ncar=len(selcars)
    SW=np.zeros((Ncar,Ncar))#within classes
    for cl in range(Ncl):
        varwithin=np.cov(classes[cl][selcars,:],ddof=0)
        SW=SW+prob[cl]*varwithin
    allclasses=classes[0][selcars,:]
    for cl in range(Ncl-1):
        allclasses=np.concatenate((allclasses,classes[cl+1][selcars,:]),axis=1)
    SM=np.cov(allclasses,ddof=0)
    SB = SM - SW
    return SW,SM,SB

def t3_selvetorial(classes,n):
    #Calcula os critérios para seleção vetorial exaustiva de n características
    #Inputs:
    # - classes = lista em que cada elemento corresponde a um array características x padrões de cada classe
    # - n = número de características para selecionar
    #Outputs:
    # - J1, J2 e J3: critérios da seleção vetorial 
    # - combs: lista com as características empregadas para cada valor de critério
    Ncar=np.size(classes[0],axis=0)
    combs=list(itertools.combinations(range(Ncar),n))
    J1=np.zeros(len(combs))
    J2=np.zeros(len(combs))
    J3=np.zeros(len(combs))
    for i in range(len(combs)): #todas combinacoes
        SW,SM,SB=t3_matrizesdeespalhamento(classes,combs[i])
        J1[i]=np.trace(SM)/np.trace(SW)
        J2[i]=np.linalg.det(SM)/np.linalg.det(SW)
        J3[i]=np.trace(np.dot(np.linalg.inv(SW),SM))/np.size(SW,axis=0) 
    return J1, J2, J3, combs

def t4_pca(dados,m):    
     #Realiza a transformação do espaço de características usando a PCA:
     #Inputs:
     # - dados= matriz L x N (caracteristicas x padroes)
     # - m = dimensão do espaço de componentes principais
     #Outputs:
     # - w = autovalores
     # - v = autovetores
     # - mse = erro quadrático médio da projeção
     # - dadosproj = matriz contendo os dados projetados
     Sigma=np.cov(dados,ddof=0)
     w,v=np.linalg.eigh(Sigma) #v: colunas são os autovetores
     comps=w.argsort()[::-1] #ordem decrescente
     comps=list(comps[0:m])
     v=v[:,comps]
     dadosproj=np.dot(v.T,dados)  
     if m==1:
         dadosproj=dadosproj[0]
     mse=100*(1-sum(w[comps])/sum(w))
     w=w[comps]
     return w,mse,dadosproj,v

def t4_svd(dados,m):    
     #Realiza a transformação do espaço de características usando a PCA:
     #Inputs:
     # - dados= matriz L x N (caracteristicas x padroes)
     # - m = dimensão do espaço de componentes principais
     #Outputs:
     # - w = autovalores
     # - v = autovetores
     # - mse = erro quadrático médio da projeção
     # - dadosproj = matriz contendo os dados projetados
    dadosnm=np.zeros_like(dados)
    for i in range(np.size(dadosnm,axis=0)):
        dadosnm[i,:]=dados[i,:]-np.mean(dados[i,:])
    U,D,VT=np.linalg.svd(dadosnm)
    w=D**2/np.size(dadosnm,axis=1)
    comps=w.argsort()[::-1] #ordem decrescente
    comps=list(comps[0:m])
    v=U[:,comps]
    dadosproj=np.dot(v.T,dados)  
    if m==1:
         dadosproj=dadosproj[0]
    mse=100*(1-sum(w[comps])/sum(w))
    w=w[comps]
    return w,mse,dadosproj,v

def gerandodadosgaussianos(medias,covariancias,N,priors,plotar=True, seed=0,angulo=[0,0]):
    # Essa funcao gera um conjunto de dados simulados representando um
    # determinado numero de caracteristicas em um determinado numero de classes. 
    # As classes possuem medias distintas e covariancias distintas. Os dados
    # seguem uma distribuicao gaussiana.
    # INPUT:
    # -medias =  classes x caracteristicas (matriz contendo as medias das 
    #    caracteristica para cada classe)
    # -covariancia =  classes x caracteristicas x caracteristicas (matrizes de 
    #    covariancia para cada classe)
    # -N = numero de padroes a serem gerados
    # -priors = array classes x 1 (prior de cada classe: probabilidade de um padrao 
    #    pertencer a cada classe), funciona tb com uma lista.
    # - plotar = True (faz grafico - 2 ou tres dimensoes), False (nao faz grafico)
    # -seed = controle do seed na geracao de dados aleatorios
    # - angulo = angulo da visualizacao em caso de plot 3d.
    # 
    # OUTPUT:
    # - dadossim=caracteristicas x padroes: dados simulados
    # - classessim= vetor contendo o numero da classe (de 0 ate C-1) de 
    #     cada padrao simulado.
    M,L=np.shape(medias)
    if np.size(covariancias,axis=0)!=M |  np.size(covariancias,axis=1)!=L | np.size(covariancias,axis=2)!=L :
        print('Erro: confira a dimensao dos seus dados de input.')
        return    
    if np.size(priors)!=M :
        print('Erro: confira a dimensao dos priors.')
        return
    if np.sum(priors)!=1 :
        print('Erro: confira os valores dos priors.')
        return
    np.random.seed(seed)      
    for i in range(M):
       Ni=np.round(priors[i]*N)
       if np.all(np.linalg.eigvals(covariancias[i]) > 0)==False :
           print('Erro: confira os valores da covariancia.')
       x=np.random.multivariate_normal(medias[i],covariancias[i],size=int(Ni)) 
       if i==0:
           dadossim=x.T
           classessim=np.zeros(int(Ni),)
       else: 
           dadossim=np.concatenate((dadossim,x.T),axis=1)
           classessim=np.concatenate((classessim,np.zeros(int(Ni),)+i),axis=0)

    if plotar: 
        if L==2: #2 caracteristicas, plot 2d
            plt.figure()
            for i in range(M):                
                plt.plot(dadossim[0,classessim==i],dadossim[1,classessim==i],'o',fillstyle='none')
            plt.xlabel('Dim 1')
            plt.ylabel('Dim 2')
            plt.show()
        elif L==3:
            plt.figure()
            ax=plt.axes(projection='3d')
            for i in range(M):                
                ax.plot(dadossim[0,classessim==i],dadossim[1,classessim==i],dadossim[2,classessim==i],'o',fillstyle='none')
            ax.view_init(angulo[0],angulo[1])
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            ax.set_zlabel('Dim 3')
            plt.show()
        else:
            print('Grafico é exibido apenas para 2 ou 3 dimensões')
    return dadossim, classessim  


def t5_classbayesgauss(medias,covariancias,priors,x):          
    #Classificador bayesiano para classes gaussianas. 
    #INPUT
    # - medias M x L, cada linha é o vetor de medias de uma clase distinta (M)
    # - covariancias ( M x L x L):  covariancias (L x L) de cada classe 
    # - priors (M x 1) priors de cada classe.
    # - x (N x L) = cada linha (N) é um padrao com L característica
    #OUTPUT
    # - probsposteriori (N x M) = probabilidade de cada padrao pertencer a cada classe
    # - classebayes (N x 1) = classe atribuida a cada padrao no classificador
    # bayesiano

    M,L=np.shape(medias)
    if np.size(covariancias,axis=0)!=M |  np.size(covariancias,axis=1)!=L | np.size(covariancias,axis=2)!=L :
        print('Erro: confira a dimensao dos seus dados de input.')
        return    
    if np.size(priors)!=M :
        print('Erro: confira a dimensao dos priors.')
        return
    if np.sum(priors)!=1 :
        print('Erro: confira os valores dos priors.')
        return
    N,L2=np.shape(x)        
    if L2!=L :
        print('Erro: confira a dimensao dos padroes.')
        return

    probsposteriori=np.zeros((N,M))
    classebayes=np.zeros(N,)
    for j in range(M): #classe
        A=np.linalg.inv(covariancias[j])
        for i in range(N): #padroes
            Norm=1/np.sqrt(((2*np.pi)**L)*np.linalg.det(covariancias[j]))
            xm=x[i]-medias[j]
            expoente=-(np.dot(xm,np.dot(A,xm.T)))/2
            probsposteriori[i,j]=priors[j]*Norm*np.exp(expoente)
    classebayes=np.argmax(probsposteriori,axis=1)
    return classebayes,probsposteriori

def t5_classdistminima(medias,covariancia,x):
    # Classificador de distância mínima
    #INPUT
    # - medias M x L, cada linha é o vetor de medias de uma clase distinta (M)
    # - covariancia ( L x L):  a covariancia comum entre as classes (L x L)
    # - x (N x L) = cada linha (N) é um padrao com L características
    #OUTPUT
    # - classeeucl (N x 1) = classificação usando a distância euclidiana
    # - classebayes (N x 1) = classificação usando a distância de mahalanobis
    M,L=np.shape(medias) #carac x classes
    if np.size(covariancia,axis=1)!=L | np.size(covariancia,axis=0)!=L :
        print('Erro: confira a dimensao dos seus dados de input.')
        return    
    N,L2=np.shape(x) #padroes x carac       
    if L2!=L :
        print('Erro: confira a dimensao dos padroes.')
        return    
    disteucl=np.zeros((N,M))
    distmaha=np.zeros((N,M))
    classeucl=np.zeros(N,)
    classmaha=np.zeros(N,)
    A=np.linalg.inv(covariancia)
    for i in range(N):
        for j in range(M):
            disteucl[i,j]=np.sqrt(np.dot((x[i]-medias[j]),(x[i]-medias[j]).T))
            distmaha[i,j]=np.sqrt(np.dot((x[i]-medias[j]),np.dot(A,(x[i]-medias[j]).T) ))
    classeucl=np.argmin(disteucl,axis=1)        
    classmaha=np.argmin(distmaha,axis=1)        
    return classeucl, classmaha

def t5_maxlikelihood(x):
    #Estima de media e covariância por Maximum Likelihood
    #x = matriz N x L (padroes x caracteristicas)
    media=np.mean(x,axis=0)
    covariancia=np.cov(x.T,ddof=0)
    return media, covariancia

def t6_perceptron(classe1, classe2, rho=0.05,maxsteps=10000):
    #Perceptron para duas classes
    #INPUTS:
    # - classe 1: L x N1 da primeira classe (características x padrões)
    # - classe 2: L x N2 da segunda classe (características x padrões)
    # - rho: learning parameter
    # - maxsteps = numero maximo de iteracoes
    #OUTPUT:
    # - ws: vetor com os pesos do classificador 
    # - hs: indica se perceptron convergiu (=0) ou não convergiu (=número de erros do perceptron)
    # - t: número de iterações realizadas
    L1,N1=np.shape(classe1)
    L2,N2=np.shape(classe2)
    if L1!=L2:
        print('ERRO: Classes devem ter o mesmo número de características')
        return
    X=np.hstack((classe1, classe2)) 
    C=np.hstack((-np.ones(N1,),np.ones(N2,)))
    X=np.vstack((X,np.ones((1,N1+N2))))    
    w=np.random.randn(L1+1,)
    ws=w;    
    def classifica(w,X,C):
        res=np.sign(np.dot(w.T,X))
        Y=X[:,res*C<=0] 
        CY=-C[res*C<=0]
        return Y, CY    
    (Y,CY)=classifica(w,X,C)
    hs=np.shape(Y)[1]
    t=0
    while ((hs>0) & (t<maxsteps)):
        for i in range(L1+1):
            w[i]=w[i]-np.sum(rho*Y[i,:]*CY)
        [Y,CY]=classifica(w,X,C)
        t=t+1;
        ht=np.shape(Y)[1]
        if ht<hs:
            hs=ht
            ws=w
    if hs==0:
        print('Perceptron convergiu')
    else:
        print('Limite de interacoes atingido')
    return ws, hs, t            

def t6_plotaperceptron(classe1,classe2,w,titulo,angulo=(-140,40)):
    #Faz o gráfico do Perceptron (para 2 ou 3 características)
    # Inputs:
    # - classe 1: L x N1 da primeira classe (características x padrões)
    # - classe 2: L x N2 da segunda classe (características x padrões)
    # - w = vetor de pesos do perceptron
    # - titulo = string com o titulo do grafico
    # - angulo = angulo de visualização (para o caso de 3 características)
    L=np.size(w) #dimensão do Perceptron (estendida, ou seja = número de características + 1)
    if L==3:
        plt.figure()
        plt.plot(classe1[0,:],classe1[1,:],'bo',fillstyle='none')
        plt.plot(classe2[0,:],classe2[1,:],'ro',fillstyle='none')
        x1=plt.xlim()[0]
        x2=plt.xlim()[1]
        x=np.linspace(x1,x2,1000)
        y=-(w[0]*x+w[2])/w[1]
        plt.plot(x,y,'--')
        plt.title(titulo)
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.show()
    elif L==4:
        plt.figure()
        ax=plt.axes(projection='3d')
        ax.plot(classe1[0,:],classe1[1,:],classe1[2,:],'bo',fillstyle='none')
        ax.plot(classe2[0,:],classe2[1,:],classe2[2,:],'ro',fillstyle='none')
        x=ax.get_xlim()
        y=ax.get_ylim()
        (xx,yy)=t8_makemeshgrid(np.asarray(x), np.asarray(y),0.01)    
        z = (-w[0]*xx -w[1]*yy -w[3])/w[2];
        ax.plot_surface(xx,yy,z,edgecolor='none')
        ax.view_init(angulo[0],angulo[1])
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.set_zlabel('Dim 3')
        plt.show()
    else:
        print('ERRO: apenas para 2 ou 3 dimensões')

def t7_LS(classe1, classe2, c=0):
    #Classificador LS para duas classes.
    #INPUTS:
    # - classe 1: L x N1 da primeira classe
    # - classe 2: L x N2 da segunda classe
    # - c: parametro de regularidade (controla a singularidade da matriz de
    # correlacao)
    #OUTPUT:
    # - w: vetor com os pesos do classificador linear (ultimo elemento é o termo não
    # homogeneo)
    L1,N1=np.shape(classe1)
    L2,N2=np.shape(classe2)
    if L1!=L2:
        print('ERRO: Classes devem ter o mesmo número de características')
        return
    X=np.hstack((classe1, classe2)) 
    X=np.vstack((X,np.ones((1,N1+N2))))    
    y=np.hstack((-np.ones(N1,),np.ones(N2,)))
    w=np.dot(np.linalg.inv(np.dot(X,X.T)+c*np.eye(L1+1)),np.dot(X,y));
    return w

def t7_FDA(classes,n):
    # Encontra a transformação para o espaço da Analise Discriminante Linear de Fisher entre classes distintas
    # INPUT:
    # - classes: lista de classes. classes[i]= (L x Ni): L caracteristicas e Ni padroes na classe i
    # - n = numero de caracterisiticas para selecionar
    # OUTPUT:
    # - A: matriz de projecao (as colunas sao os vetores de projecao para o novo espaço)
    # - Lambda: autovalores
    C=len(classes)    
    L=np.size(classes[0],axis=0)
    for i in range(C):
        if np.size(classes[i],axis=0)!=L:
            print('ERRO: Confira a dimensao das variaveis de input')
            return  
    SW,SM,SB=t3_matrizesdeespalhamento(classes,range(L))
    (Lambda,Vec)=np.linalg.eig(np.dot(np.linalg.inv(SW),SB))
    s=Lambda.argsort()
    Lambda=Lambda[s[::-1]]#ordem decrescente
    Vec=Vec[:,s[::-1]]#ordem decrescente
    A=np.real(Vec[:,0:n])
    Lambda=Lambda[0:n]
    return A,Lambda
    
def t8_makemeshgrid(x,y,h=0.02):
    # Cria um mesh para grafico de classificadores
    # INPUTS:
    # - x = dado de base para o eixo x
    # - y = dado de base para o eixo y
    # - z = dado de base para o eixo z
    # - h = passo
    # OUTPUTS:
    # - xx e yy = arrays do mesh
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def t8_plotcontours(ax,clf,xx,yy,**params):
    #Plota as superfícies de decisão de um classificador
    #INPUTS:
    # - objeto do matplotlib (eixo para usar)
    # - clf: um classificador
    # - xx e yy: outputs do meshgrid 
    # params: dicionário de parâmetros (opcionais) para a função contourf
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def t8_SVM(X,classesX,C=0.1,kernel='linear',tol=0.001,gamma='scale'):
    #Treina uma SVM em um conjunto de dados X
    # INTPUS:
    # - X = matriz padrões x características
    # - classesX = vetor com as classes dos padrões (0 ou 1)
    # - C, kernel e tol são os parâmetros da SVM 
    # OUTPUTS:
    # - clf = objeto da SVM
    # - txerro = erro de treinamento
    clf=svm.SVC(C=C,kernel=kernel,tol=tol,gamma=gamma)
    clf.fit(X,classesX)
    for i in range(len(clf.n_support_)):
        print('Numero de vetores de suporte classe '+str(i)+': '+str(clf.n_support_[i]))
    y=clf.predict(X)
    txerro=sum(y!= classesX)/classesX.size
    print('Taxa de erro treinamento: '+str(100*txerro)+'%')
    return clf, txerro
