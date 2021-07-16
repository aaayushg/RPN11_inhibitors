#!/usr/bin/env python2.7

import requests
from bs4 import BeautifulSoup
import argparse
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fetchZinc(id):
    f = 'https://zinc15.docking.org/substances/%s' % id
    r = requests.get(f, timeout = 10, verify=False) # ...timeout after 10 sec

    v = {}

    # Fetch data only if web page responds correctly...
    if r.status_code == 200:
    	c = r.content
    	s = BeautifulSoup(c, features='lxml')
    	t = s.findAll('table') # ...All tables

    	v['ID']   = id
    	v['MW']   = float(t[0].findAll('td')[3].string)
    	v['logP'] = float(t[0].findAll('td')[4].string)
    	v['Ring'] =   int(t[1].findAll('td')[1].string)
    	v['HBD']  =   int(t[2].find('td', {'title':"Hydrogen Bond Donors"}).string)
    	v['HBA']  =   int(t[2].find('td', {'title':"Hydrogen Bond Acceptors"}).string)
    	v['RB']   =   int(t[2].find('td', {'title':"Rotatable Bonds"}).string)
    	v['tPSA'] =   int(t[2].find('td', {'title':"Polar Surface Area"}).string)
    	return v

fo   = 'data_48.dat'
idsfile = 'ids-48.csv'

with open(fo,'w') as fh:
    fh.write("%s , %-8s , %s , %s , %s , %s , %s, %s\n" % ( 'ID',
                                                    'MW',
                                                    'logP',
                                                    'Ring',
                                                    'HBD',
                                                    'HBA',
                                                    'RB' ,
                                                    'tPSA'))


    v = {}
    with open(idsfile, 'r') as rh:
	for i in rh:
		try:
        		v = fetchZinc(i)
        		fh.write(i[0:12]+",   " + "%6.2f , %6.2f , %3d , %3d , %3d , %3d, %3d\n" % (
                                                                    v['MW'],
                                                                    v['logP'],
                                                                    v['Ring'],
                                                                    v['HBD'] ,
                                                                    v['HBA'] ,
                                                                    v['RB']  ,
                                                                    v['tPSA']))
        		fh.flush()
		except:
			pass


