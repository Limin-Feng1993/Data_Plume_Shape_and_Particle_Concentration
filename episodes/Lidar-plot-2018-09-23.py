#coding=utf-8
#plot lidar color map
#by Limin Feng
import netCDF4
import h5py
import shapefile
import netCDF4 as nc 
from netCDF4 import Dataset
import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from netCDF4 import Dataset
import numpy
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator
import os
import pandas as pd
import datetime
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.dates import AutoDateLocator, DateFormatter
import pathlib
from pathlib import Path
import time
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
import math
import matplotlib as mpl

# datetime时间转为字符串
def Changestr(datetime1):
    str1 = datetime1.strftime('%Y-%m-%d %H:%M')
    return str1

# 字符串时间转为时间戳
def Changetime(str1):
    Unixtime = time.mktime(time.strptime(str1, '%Y-%m-%d %H:%M'))
    return Unixtime

# datetime时间转为时间戳
def Changestamp(dt1):
    Unixtime = time.mktime(time.strptime(dt1.strftime('%Y-%m-%d %H:%M'), '%Y-%m-%d %H:%M'))
    return Unixtime

# 时间戳转为datetime时间
def Changedatetime(timestamp):
    dt = datetime.fromtimestamp(timestamp)
    return dt

# uinx时间戳转换为本地时间
def Localtime(datetime1):
    Localtime = time.strftime('%Y-%m-%d %H:%M',time.localtime(datetime1))
    return Localtime
    
# 字符串时间转换为计算机存储时间
def Normaltime(datetime1):
    #Normaltime = datetime.strptime(datetime1,'%Y-%m-%d %H:%M')
    Normaltime = datetime.strptime(datetime1, '%Y%m%d%H%M')
    return Normaltime 
     
def Normaltime1(datetime1):
    Normaltime = datetime.strptime(datetime1,'%Y-%m-%d %H:%M')
    return Normaltime  

# 转置矩阵    
def trans(m):
    a = [[] for i in m[0]]
    for i in m:
        for j in range(len(i)):
            a[j].append(i[j])
    return a 

###########start 获取文件路径、文件名、后缀名############
def get_filename(filename):
  (filepath,tempfilename) = os.path.split(filename);
  (shotname,extension) = os.path.splitext(tempfilename);
  #return filepath,shotname,extension
  return shotname
#########end 获取文件路径、文件名、后缀名############

#Excel时间序列处理方法一
def TS(x):
    return (x - np.datetime64('1970-01-01T00:00:00Z'))/np.timedelta64(1, 's')
    #return datetime.utcfromtimestamp(x.astype('O')/1e9)
    #return datetime.fromtimestamp(x.tolist()/1e9)

##Excel时间序列处理方法二
def DT(x):
    return datetime.utcfromtimestamp(x)
    
plt.rcParams['axes.unicode_minus']=False 
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Times New Roman'#中文除外的设置成New Roman，中文设置成宋体,NSimSun,
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Times New Roman'#中文除外的设置成New Roman，中文设置成宋体,NSimSun,

path=pathlib.Path('C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\Python36\\Scripts\\Sky_recog_pred\\small data\\')
df=path/'2018-09 UVT.xlsx'
U=pd.DataFrame(pd.read_excel(df, sheet_name=u'U'))
V=pd.DataFrame(pd.read_excel(df, sheet_name=u'V'))
TH=pd.DataFrame(pd.read_excel(df, sheet_name=u'TH'))

X2=TH['Time']
H3=TH['Height']
X3=[DT(TS(i)) for i in X2]

X4, Y4= np.meshgrid(X3,H3)

fp='sphere-09-23.txt'
ft='dust-09-23.txt'
Ydata=np.loadtxt('Height.txt')#200行*1列 
lie=[]


with open(fp,'r') as f0:
	for line in f0:
		#line=line.split("\t")
		line=line.split()
		lie.append(line[0])
print (lie)
Xdata=[]
for i in lie:
	X= Normaltime(i)
	Xdata.append(X)
print (Xdata)

Zdata=[]
with open(fp,'r') as f1:
	for line in f1:
		#line=line.split("\t")
		line=line.split()
		for flt in line:
			flt=float(flt)
		Zdata.append(line[1:201])

Zdata=np.array(Zdata,dtype=np.float)
Zdata=trans(Zdata)#200行*288列#Zdata=np.transpose(Zdata)
Zdata=np.array(Zdata,dtype=np.float)

#Zdata=np.where(Zdata>=1, 0.001, Zdata)
Zdata=np.where(Zdata>=0.175, 0.175, Zdata)
#Zdata=np.where(Zdata>=0.45, 0.45, Zdata)

Zdata=np.where(Zdata==-999000, 1000, Zdata)
Zdata=np.where(Zdata<0, 0.001, Zdata)
#Zdata=1000*Zdata

print(Zdata)

fig = plt.figure(figsize=(16,16)) #dpi=300,  facecolor="white"      
#plt.figure(figsize=(16, 8))                                                             
#ax = fig.add_subplot(111)
#fig.subplots_adjust(bottom=0.025, left=0.025, top = 0.975, right=0.975)
plt.subplot(211)
font = FontProperties(fname=r"C:\\Windows\\Fonts\\times.ttf")   

X,Y = np.meshgrid(Xdata,Ydata)
minval,maxval=0, 0.2
#minval,maxval=0, 1.0
#minval,maxval=0, 0.5
#norm = matplotlib.colors.Normalize(vmin=0, vmax=500, clip=False)
cs = plt.contourf(X,Y,Zdata, np.arange(minval, maxval, 0.001),cmap=plt.cm.get_cmap('jet'))#norm=norm, offset=-2, shrink=.92
plt.barbs(X4, Y4, U, V,barbcolor='white', flagcolor='r',linewidth=3, length=8, pivot='middle')      
dateS=datetime(2018, 9, 23, 6, 00)
dateE=datetime(2018, 9, 24, 18, 00)
plt.xlim(dateS, dateE)                                                                                                  
#设置时间标签显示格式
plt.gca()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))  
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=12))  #設置x軸主刻度間距
plt.yticks(np.linspace(500,3000,6, endpoint=True))
#plt.yticks(np.logspace(500,3000,6, base=10))
#ax.set_yticklabels( ('500', '1000', '1500', '2000', '2500', '3000'), fontproperties=font,fontsize=18)

cbar = plt.colorbar(cs)
cbar.set_label('Extinction Coefficient',fontproperties=font,fontsize=24)
cbar.set_ticks(np.linspace(0, 0.2,5))
#cbar.set_ticks(np.linspace(0, 1.0,5))
#cbar.set_ticks(np.linspace(0, 0.5, 6))
#cbar.set_ticklabels(('0.0', '0.1', '0.2', '0.3',  '0.4',  '0.5',  '0.6', '0.7', '0.8', '0.9',  '1.0')) 
cbar.ax.tick_params(labelsize=24)
cs.ax.tick_params(labelsize=24)#plt.xticks(fontsize=18)

#ax.set_xlabel('Time', fontproperties=font,fontsize=18) 
plt.ylabel('Height (m)',fontproperties=font,fontsize=24)
plt.ylim(100, 3000)
#plt.clim(0,500)
#ax.legend(..., fontsize=20)

titleStr='532 nm Sphere Extinction Coefficient in Rizhao'
plt.title(titleStr,fontproperties=font,fontsize=28)
plt.tight_layout()
#plt.ax_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8，hspace=0.2, wspace=0.3)
#plt.margins(0,0)
#plt.savefig(str(fp)+'.png')
#plt.clf()
##############################################################################
lie=[]
with open(ft,'r') as f0:
	for line in f0:
		#line=line.split("\t")
		line=line.split()
		lie.append(line[0])
print (lie)
Xdata=[]
for i in lie:
	X= Normaltime(i)
	Xdata.append(X)
print (Xdata)

Zdata=[]
with open(ft,'r') as f1:
	for line in f1:
		#line=line.split("\t")
		line=line.split()
		for flt in line:
			flt=float(flt)
		Zdata.append(line[1:201])

Zdata=np.array(Zdata,dtype=np.float)
Zdata=trans(Zdata)#200行*288列#Zdata=np.transpose(Zdata)
Zdata=np.array(Zdata,dtype=np.float)

#Zdata=np.where(Zdata>=1, 0.001, Zdata)
Zdata=np.where(Zdata>=0.175, 0.175, Zdata)
#Zdata=np.where(Zdata>=0.49, 0.49, Zdata)

Zdata=np.where(Zdata==-999000, 1000, Zdata)
Zdata=np.where(Zdata<0, 0.001, Zdata)
#Zdata=1000*Zdata

print(Zdata)

#fig = plt.figure(figsize=(16,8)) #dpi=300,  facecolor="white"                                                                   
#ax = fig.add_subplot(111)
#plt.figure(figsize=(16, 8))
#fig.subplots_adjust(bottom=0.025, left=0.025, top = 0.975, right=0.975)
plt.subplot(212)
font = FontProperties(fname=r"C:\\Windows\\Fonts\\times.ttf")   

X,Y = np.meshgrid(Xdata,Ydata)
minval,maxval=0, 0.2
#minval,maxval=0, 1.0
#minval,maxval=0, 0.5
#norm = matplotlib.colors.Normalize(vmin=0, vmax=200, clip=False)

cs = plt.contourf(X,Y,Zdata,np.arange(minval,maxval, 0.001),  cmap=plt.cm.get_cmap('jet'))#norm=norm,shrink=.92,offset=-2,
#plt.quiver(X4, Y4, U, V,color='white', width=0.005,scale=170)                                                                                                          
plt.barbs(X4, Y4, U, V,barbcolor='white', flagcolor='r',linewidth=3, length=8, pivot='middle')      
dateS=datetime(2018, 9, 23, 6, 00)
dateE=datetime(2018, 9, 24, 18, 00)
plt.xlim(dateS, dateE)
plt.gca()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))  
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=12))  #設置x軸主刻度間距
plt.yticks(np.linspace(500,3000,6))

#plt.yticks(np.logspace(500,3000,6, base=10), (100, 500, 1000, 2000, 3000))
#ax.set_yticklabels( ('500', '1000', '1500', '2000', '2500', '3000'), fontproperties=font,fontsize=18)

cbar = plt.colorbar(cs)
cbar.set_label('Extinction Coefficient',fontproperties=font,fontsize=24)
cbar.set_ticks(np.linspace(0, 0.2,5))
#cbar.set_ticks(np.linspace(0, 1.0,5))
#cbar.set_ticks(np.linspace(0, 0.5, 6))
#cbar.set_ticklabels(('0.0', '0.1', '0.2', '0.3',  '0.4',  '0.5',  '0.6', '0.7', '0.8', '0.9',  '1.0')) 
cbar.ax.tick_params(labelsize=24)
cs.ax.tick_params(labelsize=24)#plt.xticks(fontsize=18)

#ax.set_xlabel('Time', fontproperties=font,fontsize=18) 
plt.ylabel('Height (m)',fontproperties=font,fontsize=24)
plt.ylim(100, 3000)
#plt.xlim(X.min() * 1.1, X.max() * 1.1)
#plt.clim(0,500)
#ax.legend(..., fontsize=20)

titleStr='532 nm Non-Sphere Extinction Coefficient in Rizhao'
plt.title(titleStr,fontproperties=font,fontsize=28)
plt.tight_layout()
#plt.margins(0,0)
#plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8，hspace=0.2, wspace=0.3)
filename=get_filename(fp)

plt.savefig(str(filename)+'.png')
plt.clf()
print("ALL -> Finished OK")

