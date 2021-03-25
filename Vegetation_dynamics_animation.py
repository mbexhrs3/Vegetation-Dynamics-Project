# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 13:11:50 2014

@author: acrnrrs
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#os.system ("ctem_comp_plot.sh 22_30")

#os.system ("ctem_comp_plot_Mosaic.sh 22_30")  #MOSAIC

#os.system ("ctem_comp_plot_Composit_Mosaic.sh 22_30")  #MOSAIC AND COMPOSIT

########################### PLANT FUNCTIONAL TYPE FRACTION ###################

#fig=plt.figure(1,figsize=(9,4.5))
#ax = fig.add_subplot(111)
#fig.subplots_adjust(right=0.70)
#ax=plt.gca()




statinfo = os.stat('/HOME/rrs/VF_2016/Rudra_New/animation/Fract_PFT1.txt')
file_size=statinfo.st_size
if file_size > 250 :  #250KB size I think
  f=open('/HOME/rrs/VF_2016/Rudra_New/animation/Fract_PFT1.txt','r')
  dat_pft1 = np.genfromtxt(f)
  dat_yr=dat_pft1[:,0]
  dat_pft_1=dat_pft1[:,21]
  num_year=np.linspace(1901,2300,400)
  pft_fra1=np.vstack((dat_yr,dat_pft_1))
  pft_fra=np.transpose(pft_fra1)
  df=pd.DataFrame(pft_fra[:,1],index=[pft_fra[:,0]])
  df2=df.reindex([num_year])
  df3=np.squeeze(df2) #this is series type data
  df4=df2.transpose()  # transpose data frame 
  lst40=df4.iloc[0,360:400] #slice last 40 years data
  m40_f1="%.2f" % round(lst40.mean(axis=0),2)
  str_leg1 = ''.join(['NDL-EVG (', m40_f1, ')']) 
  p1=df4  #for animation 
  p1[np.isnan(p1)] = 0  #make nan zero

else:
 print "No PFT1 exist"
 m40_f1=0
 p1=np.zeros(400)

statinfo = os.stat('/HOME/rrs/VF_2016/Rudra_New/animation/Fract_PFT2.txt')
file_size=statinfo.st_size
if file_size > 250 :
  f2=open('/HOME/rrs/VF_2016/Rudra_New/animation/Fract_PFT2.txt','r')
  dat_pft2 = np.genfromtxt(f2)
  dat_yr=dat_pft2[:,0]
  dat_pft_2=dat_pft2[:,21]
  num_year=np.linspace(1901,2300,400)
  pft_fra1=np.vstack((dat_yr,dat_pft_2))
  pft_fra=np.transpose(pft_fra1)
  df=pd.DataFrame(pft_fra[:,1],index=[pft_fra[:,0]])
  df2=df.reindex([num_year])
  df3=np.squeeze(df2)
  df4=df2.transpose()  # transpose data frame 
  lst40=df4.iloc[0,360:400] #slice last 40 years data
  m40_f2="%.2f" % round(lst40.mean(axis=0),2)
  str_leg2 = ''.join(['NDL-DCD (', m40_f2, ')'])  
  p2=df4 
  p2[np.isnan(p2)] = 0  
else:
  print "No PFT2 exist" 
  m40_f2=0
  p2=np.zeros(400)

statinfo = os.stat('/HOME/rrs/VF_2016/Rudra_New/animation/Fract_PFT3.txt')
file_size=statinfo.st_size
if file_size > 250 :
  f3=open('/HOME/rrs/VF_2016/Rudra_New/animation/Fract_PFT3.txt','r')
  dat_pft3 = np.genfromtxt(f3)
  dat_yr=dat_pft3[:,0]
  dat_pft_3=dat_pft3[:,21]
  num_year=np.linspace(1901,2300,400)
  pft_fra1=np.vstack((dat_yr,dat_pft_3))
  pft_fra=np.transpose(pft_fra1)
  df=pd.DataFrame(pft_fra[:,1],index=[pft_fra[:,0]])
  df2=df.reindex([num_year])
  df3=np.squeeze(df2)
  df4=df2.transpose()  # transpose data frame 
  lst40=df4.iloc[0,360:400] #slice last 40 years data
  m40_f3="%.2f" % round(lst40.mean(axis=0),2)
  str_leg3 = ''.join(['BDL-EVG (', m40_f3, ')']) 
  p3=df4  
  p3[np.isnan(p3)] = 0  
else:
  print "No PFT3 exist" 
  m40_f3=0
  p3=np.zeros(400)

statinfo = os.stat('/HOME/rrs/VF_2016/Rudra_New/animation/Fract_PFT4.txt')
file_size=statinfo.st_size
if file_size > 250 :
  f4=open('/HOME/rrs/VF_2016/Rudra_New/animation/Fract_PFT4.txt','r')
  dat_pft4 = np.genfromtxt(f4)
  dat_yr=dat_pft4[:,0]
  dat_pft_4=dat_pft4[:,21]
  num_year=np.linspace(1901,2300,400)
  pft_fra1=np.vstack((dat_yr,dat_pft_4))
  pft_fra=np.transpose(pft_fra1)
  df=pd.DataFrame(pft_fra[:,1],index=[pft_fra[:,0]])
  df2=df.reindex([num_year])
  df3=np.squeeze(df2)
  df4=df2.transpose()  # transpose data frame 
  lst40=df4.iloc[0,360:400] #slice last 40 years data
  m40_f4="%.2f" % round(lst40.mean(axis=0),2)
  str_leg4 = ''.join(['BDL-DCD-CLD (', m40_f4, ')']) 
  p4=df4
  p4[np.isnan(p4)] = 0  
else:
  print "No PFT4 exist" 
  m40_f4=0
  p4=np.zeros(400)

statinfo = os.stat('/HOME/rrs/VF_2016/Rudra_New/animation/Fract_PFT5.txt')
file_size=statinfo.st_size
if file_size > 250 :
  f5=open('/HOME/rrs/VF_2016/Rudra_New/animation/Fract_PFT5.txt','r')
  dat_pft5 = np.genfromtxt(f5)
  dat_yr=dat_pft5[:,0]
  dat_pft_5=dat_pft5[:,21]
  num_year=np.linspace(1901,2300,400)
  pft_fra1=np.vstack((dat_yr,dat_pft_5))
  pft_fra=np.transpose(pft_fra1)
  df=pd.DataFrame(pft_fra[:,1],index=[pft_fra[:,0]])
  df2=df.reindex([num_year])
  df3=np.squeeze(df2)
  df4=df2.transpose()  # transpose data frame 
  lst40=df4.iloc[0,360:400] #slice last 40 years data
  m40_f5="%.2f" % round(lst40.mean(axis=0),2)
  str_leg5 = ''.join(['BDL-DCD-DRY (', m40_f5, ')']) 
  p5=df4
  p5[np.isnan(p5)] = 0  
else:
  print "No PFT5 exist" 
  m40_f5=0
  p5=np.zeros(400)

statinfo = os.stat('/HOME/rrs/VF_2016/Rudra_New/animation/Fract_PFT6.txt')
file_size=statinfo.st_size
if file_size > 250 :
  f6=open('/HOME/rrs/VF_2016/Rudra_New/animation/Fract_PFT6.txt','r')
  dat_pft6 = np.genfromtxt(f6)
  dat_yr=dat_pft6[:,0]
  dat_pft_6=dat_pft6[:,21]
  num_year=np.linspace(1901,2300,400)
  pft_fra1=np.vstack((dat_yr,dat_pft_6))
  pft_fra=np.transpose(pft_fra1)
  df=pd.DataFrame(pft_fra[:,1],index=[pft_fra[:,0]])
  df2=df.reindex([num_year])
  df3=np.squeeze(df2)
  df4=df2.transpose()  # transpose data frame 
  lst40=df4.iloc[0,360:400] #slice last 40 years data
  m40_f6="%.2f" % round(lst40.mean(axis=0),2)
  str_leg6 = ''.join(['CROP-C3 (', m40_f6, ')']) 
  p6=df4
  p6[np.isnan(p6)] = 0
else:
  print "No PFT6 exist"
  m40_f6=0
  p6=np.zeros(400)

statinfo = os.stat('/HOME/rrs/VF_2016/Rudra_New/animation/Fract_PFT7.txt')
file_size=statinfo.st_size
if file_size > 250 :
  f7=open('/HOME/rrs/VF_2016/Rudra_New/animation/Fract_PFT7.txt','r')
  dat_pft7 = np.genfromtxt(f7)
  dat_yr=dat_pft7[:,0]
  dat_pft_7=dat_pft7[:,21]
  num_year=np.linspace(1901,2300,400)
  pft_fra1=np.vstack((dat_yr,dat_pft_7))
  pft_fra=np.transpose(pft_fra1)
  df=pd.DataFrame(pft_fra[:,1],index=[pft_fra[:,0]])
  df2=df.reindex([num_year])
  df3=np.squeeze(df2)
  df4=df2.transpose()  # transpose data frame 
  lst40=df4.iloc[0,360:400] #slice last 40 years data
  m40_f7="%.2f" % round(lst40.mean(axis=0),2)
  str_leg7 = ''.join(['CROP-C4 (', m40_f7, ')']) 
  p7=df4
  p7[np.isnan(p7)] = 0
else:
  print "No PFT7 exist"
  m40_f7=0
  p7=np.zeros(400)
  
  
statinfo = os.stat('/HOME/rrs/VF_2016/Rudra_New/animation/Fract_PFT8.txt')
file_size=statinfo.st_size
if file_size > 250 :
  f8=open('/HOME/rrs/VF_2016/Rudra_New/animation/Fract_PFT8.txt','r')
  dat_pft8 = np.genfromtxt(f8)
  dat_yr=dat_pft8[:,0]
  dat_pft_8=dat_pft8[:,21]
  num_year=np.linspace(1901,2300,400)
  pft_fra1=np.vstack((dat_yr,dat_pft_8))
  pft_fra=np.transpose(pft_fra1)
  df=pd.DataFrame(pft_fra[:,1],index=[pft_fra[:,0]])
  df2=df.reindex([num_year])
  df3=np.squeeze(df2)
  df4=df2.transpose()  # transpose data frame 
  lst40=df4.iloc[0,360:400] #slice last 40 years data
  m40_f8="%.2f" % round(lst40.mean(axis=0),2)
  str_leg8 = ''.join(['GRASS-C3 (', m40_f8, ')']) 
  p8=df4
  p8[np.isnan(p8)] = 0
  
else:
  print "No PFT8 exist"
  m40_f8=0
  p8=np.zeros(400)

statinfo = os.stat('/HOME/rrs/VF_2016/Rudra_New/animation/Fract_PFT9.txt')
file_size=statinfo.st_size
if file_size > 250 :    #pft 9 only has single line output causes the problem 
  f9=open('/HOME/rrs/VF_2016/Rudra_New/animation/Fract_PFT9.txt','r')
  dat_pft9 = np.genfromtxt(f9)
  dat_yr=dat_pft9[:,0]
  dat_pft_9=dat_pft9[:,21]
  num_year=np.linspace(1901,2300,400)
  pft_fra1=np.vstack((dat_yr,dat_pft_9))
  pft_fra=np.transpose(pft_fra1)
  df=pd.DataFrame(pft_fra[:,1],index=[pft_fra[:,0]])
  df2=df.reindex([num_year])
  df3=np.squeeze(df2)
  df4=df2.transpose()  # transpose data frame 
  lst40=df4.iloc[0,360:400] #slice last 40 years data
  m40_f9="%.2f" % round(lst40.mean(axis=0),2)
  str_leg9 = ''.join(['GRASS-C4 (', m40_f9, ')']) 
  p9=df4
  p9[np.isnan(p9)] = 0
else:
  print "No PFT9 exist"
  m40_f9=0  
  p9=np.zeros(400)
  
statinfo = os.stat('/HOME/rrs/VF_2016/Rudra_New/animation/Fract_PFT10.txt')
file_size=statinfo.st_size
if file_size > 250 :
  f10=open('/HOME/rrs/VF_2016/Rudra_New/animation/Fract_PFT10.txt','r')
  dat_pft10 = np.genfromtxt(f10)
  dat_yr=dat_pft10[:,0]
  dat_pft_10=dat_pft10[:,21]
  num_year=np.linspace(1901,2300,400)
  pft_fra1=np.vstack((dat_yr,dat_pft_10))
  pft_fra=np.transpose(pft_fra1)
  df=pd.DataFrame(pft_fra[:,1],index=[pft_fra[:,0]])
  df2=df.reindex([num_year])
  df3=np.squeeze(df2)
  df4=df2.transpose()  # transpose data frame 
  lst40=df4.iloc[0,360:400] #slice last 40 years data
  m40_f10="%.2f" % round(lst40.mean(axis=0),2)
  str_leg10 = ''.join(['BARE (', m40_f10, ')']) 
  p10=df4
  p10[np.isnan(p10)] = 0
else:
  print "No PFT10 exist"
  m40_f10=0
  p10=np.zeros(400)

  

############# pft animation 
numyear=np.linspace(1,400,400)
year=numyear
#fraction=np.vstack([year,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10])
fraction=np.vstack([year,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10])
F=np.transpose(fraction)


fig=plt.figure(50,figsize=(9,7))
#plt.subplot(211)
ax = fig.add_subplot(111)
fig.subplots_adjust(right=0.70)
ax=plt.gca()

plt.axis('equal')
plt.xticks(())
plt.yticks()

#labels = 'NDL-EVG', 'NDL-DCD', 'BDL-EVG', 'BDL-DCD-CLD', 'BDL-DCD-DRY', 'C-C3', 'C-C4','G-C3','G-C4','BARE'

labels =  'CROP-C3', 'CROP-C4','NDL-EVG', 'NDL-DCD', 'BDL-EVG', 'BDL-DCD-CLD', 'BDL-DCD-DRY', 'GRASS-C3','GRASS-C4','BARE'

ims = []

def fracs(a6,a7,a1,a2,a3,a4,a5,a8,a9,a10):
    return [a6,a7,a1,a2,a3,a4,a5,a8,a9,a10]

def Tit(t):
   return (plt.title(t))


for line in F:
    Y =  int(line[0])
  
    a1 = line[1]*100
    a2 = line[2]*100
    a3 = line[3]*100  
    a4 = line[4]*100  
    a5 = line[5]*100
    a6 = line[6]*100
    a7 = line[7]*100      
    a8 = line[8]*100  
    a9 = line[9]*100
    a10 = line[10]*100
  
    explode=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    im=plt.pie(fracs(a6,a7,a1,a2,a3,a4,a5,a8,a9,a10), startangle=90, radius=1.2,\
    colors=('chocolate', 'cornflowerblue','r', 'orange', 'b', 'orchid', 'm',   'limegreen', 'darkgreen','gray'))
    

#    plt.legend(labels, loc='right', shadow=True)
    
    txt=plt.text(0.67, 1.05,Y, horizontalalignment='center',verticalalignment='center', transform=ax.transAxes,fontsize=25,\
    fontweight='bold',color='k')
    
    ax.legend(labels,loc='center left', fontsize='medium',bbox_to_anchor=(1.05, 0.5),\
    fancybox=True, shadow=True, ncol=1,borderaxespad=0.)
#    im3 =tit2(t)
    
    ims.append( [txt] + im[0] + im[1])


ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=False, repeat_delay=1000)

plt.text(0.37, 1.05,'YEAR  = ', horizontalalignment='center',verticalalignment='center', transform=ax.transAxes,fontsize=25,
fontweight='bold',color='k')


#ani.save('India_composite.avi',dpi=100)
ani.save('animation.mp4',dpi=100)
plt.show()


#######################################################################################################################
#######################################################################################################################

    


#os.system ("rm -f *.png")

#os.system ("rm -f *.txt")








