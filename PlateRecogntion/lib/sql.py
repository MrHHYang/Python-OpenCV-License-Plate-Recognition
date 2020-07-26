#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tkinter
import pymysql

def sql(TIME, COLOR1, TEXT1, SOURCE):
    """连接数据库，将数据插入数据表"""
    try:
        # 打开数据库连接
        db = pymysql.connect(host="localhost",
                             port=3306,
                             user="root",
                             passwd="123",
                             database="chepai",
                             charset="utf8")
    except:
        print("数据库连接失败")
        return
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()

    # SQL 插入语句
    sql = "INSERT INTO ALLVEHICLE(TIME, \
       COLOR1, TEXT1, SOURCE) \
       VALUES ('%s', '%s', '%s', '%s')" % \
        (TIME, COLOR1, TEXT1, SOURCE)

    try:
        # 执行sql语句
        cursor.execute(sql)
        # 提交到数据库执行
        db.commit()
        print("数据库写入成功")
    except:
        # 如果发生错误则回滚
        db.rollback()
        print("数据库写入失败")

    sql0 = "SELECT * FROM BLACKLIST WHERE TEXT1 like ('%s')" \
           % (TEXT1)
    try:
        # 执行sql语句
        len = cursor.execute(sql0)
        # 提交到数据库执行
        db.commit()
        if len == 0:
            tkinter.messagebox.showwarning(title='提示', message='快速放行！')
        else:
            tkinter.messagebox.showwarning(title='提示', message='拦截此车辆！')
    except:
        # 如果发生错误则回滚
        db.rollback()
        print("数据库查询失败")

    # 关闭数据库连接
    db.close()

def sql2(TIME, COLOR1, TEXT1, SOURCE):
    """连接数据库，将数据插入数据表"""
    try:
        # 打开数据库连接
        db = pymysql.connect(host="localhost",
                             port=3306,
                             user="root",
                             passwd="123",
                             database="chepai",
                             charset="utf8")
    except:
        print("数据库连接失败")
        return
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()

    # SQL 插入语句
    sql = "INSERT INTO BLACKLIST(TIME, \
       COLOR1, TEXT1, SOURCE) \
       VALUES ('%s', '%s', '%s', '%s')" % \
        (TIME, COLOR1, TEXT1, SOURCE)

    try:
        # 执行sql语句
        cursor.execute(sql)
        # 提交到数据库执行
        db.commit()
        print("数据库写入成功")
    except:
        # 如果发生错误则回滚
        db.rollback()
        print("数据库写入失败")

    # 关闭数据库连接
    db.close()


def create_sql():
    # 创建表CARINFO，车牌信息的表
    try:
        # 打开数据库连接
        db = pymysql.connect(host="localhost", port=3306, user="root", passwd="123", database="chepai", charset="utf8")
    except:
        print("数据库连接失败")
        return
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    # 使用预处理语句创建表
    sql = """CREATE TABLE ALLVEHICLE (
            TIME VARCHAR(100),
            COLOR1 VARCHAR(100), 
            TEXT1 VARCHAR(100), 
            SOURCE VARCHAR(500))"""

    try:
        # 执行sql语句
        cursor.execute(sql)
        # 提交到数据库执行
        db.commit()
        print("数据库创建成功")
    except:
        # 如果发生错误则回滚
        db.rollback()
        print("数据库已存在")

    # 使用预处理语句创建表
    sql2 = """CREATE TABLE BLACKLIST (
            TIME VARCHAR(100),
            COLOR1 VARCHAR(100), 
            TEXT1 VARCHAR(100), 
            SOURCE VARCHAR(500))"""

    try:
        # 执行sql语句
        cursor.execute(sql2)
        # 提交到数据库执行
        db.commit()
        print("数据库创建成功")
    except:
        # 如果发生错误则回滚
        db.rollback()
        print("数据库已存在")


    # 关闭数据库连接
    db.close()
