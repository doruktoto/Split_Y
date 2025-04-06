#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:11:04 2019

@author: leandro
"""
import json

currentStatus = {}
statusFilename = "processInfo.json"

def setStatusFilename(path):
    global statusFilename
    statusFilename = path

def saveStatus(key, value):
    global currentStatus

    if not key in currentStatus or type(currentStatus[key]) is not list :
        currentStatus[key] = value
    else:
        currentStatus[key] += value

    with open(statusFilename, "w+") as f:
        json.dump(currentStatus, f)
