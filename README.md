# Detect-Heavy-Drinking-with-Accelerometer-Data
Northeastern Graduate Class - DS 5500, Capstone Project

## Motivation:
Alcoholic beverages are prevalent in the current societal landscape and are often associated with socializing.
With alcohol becoming a routine part of life, the risks associated with alcohol abuse or alcoholism have
increased. Excessive drinking, alcoholism contributes to nearly 5.3% of deaths worldwide according to World
the population, meet the diagnostic criteria for alcohol abuse or alcoholism.

Just-in-time adaptive interventions (JITAIs) delivered on mobile applications have shown promising effectiveness in promoting healthier drinking habits, however, current methods lack generalizability or use certain personal information which could violate users’ privacy. This creates a need for targeted and accurate intervention amongst people during heavy drinking episodes. With the surge in the more accurate data collected through smartphones, the data garnered through accelerometers of these devices can help identify patterns indicating heavy drinking episodes or alcohol abuse. The project aims to deliver a machine learning tool to analyze this complex dataset, which can drive the necessity of targeted and anonymous mobile-based interventions.

## Objective:
Given mobile accelerometer data over a given window of time, identify if the user is intoxicated or sober.
The above goal will be acheived by building classification models by analyzing the intoxication levels using Transdermal alcohol content (TAC)[later converted to binary labels] and triaxial accelerometer readings.

## Data:
Our project uses the open-source data: Bar Crawl: Detecting Heavy Drinking Data Set in UCI Machine Learning Repository.
The dataset consists of 715 TAC and 14,057,567 accelerometer readings from 13 participants – 11 iPhones and 2 android phones. 
