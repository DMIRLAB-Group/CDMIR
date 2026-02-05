# -*- coding: utf-8 -*-
import logging

class Logger:
    """[
        logger：提供日志接口，供应用代码使用。logger最长用的操作有两类：配置和发送日志消息。可以通过logging.getLogger(name)获取logger对象，
                如果不指定name则返回root对象，多次使用相同的name调用getLogger方法返回同一个logger对象。
        handler：将日志记录（log record）发送到合适的目的地（destination），比如文件，socket等。一个logger对象可以通过addHandler方法添加0
                到多个handler，每个handler又可以定义不同日志级别，以实现日志分级过滤显示。
        filter：提供一种优雅的方式决定一个日志记录是否发送到handler。
        formatter：指定日志记录输出的具体格式。formatter的构造方法需要两个参数：消息的格式字符串和日期字符串，这两个参数都是可选的。
                    %(name)s Logger的名字
                    %(levelname)s 文本形式的日志级别
                    %(message)s 用户输出的消息
                    %(asctime)s 字符串形式的当前时间。默认格式是 “2003-07-08 16:49:45,896”。逗号后面的是毫秒
                    %(levelno)s 数字形式的日志级别
                    %(pathname)s 调用日志输出函数的模块的完整路径名
                    %(filename)s 调用日志输出函数的模块的文件名
                    %(module)s  调用日志输出函数的模块名
                    %(funcName)s 调用日志输出函数的函数名
                    %(lineno)d 调用日志输出函数的语句所在的代码行
                    %(created)f 当前时间，用UNIX标准的表示时间的浮点数表示
                    %(relativeCreated)d 输出日志信息时的，自Logger创建以来的毫秒数
                    %(thread)d 线程ID
                    %(threadName)s 线程名
                    %(process)d 进程ID  
        ]
    """
    def __init__(self, path, clevel = logging.DEBUG, Flevel = logging.DEBUG, CMD_render=False):
        # 初始化，getLogger()方法后面最好加上所要日志记录的模块名字，后面的日志格式中的%(name)s 对应的是这里的模块名字
        self.logger = logging.getLogger(path)
        # 设置级别,Logging中有NOTSET < DEBUG < INFO < WARNING < ERROR < CRITICAL这几种级别，日志会记录设置级别以上的日志
        self.logger.setLevel(logging.DEBUG)
        self.fmt = logging.Formatter('%(message)s')
        if CMD_render:
            #设置CMD日志, StreamHandler
            sh = logging.StreamHandler()
            sh.setFormatter(self.fmt)
            sh.setLevel(clevel)
            self.logger.addHandler(sh)
        #设置文件日志, FileHandler
        fh = logging.FileHandler(path, mode='a')
        fh.setFormatter(self.fmt)
        fh.setLevel(Flevel)
        self.logger.addHandler(fh)

    def debug(self,message):
        self.logger.debug(message)

    def info(self,message):
        self.logger.info(message)
  
    def warning(self,message):
        self.logger.warning(message)

    def error(self,message):
        self.logger.error(message)

    def critical(self,message):
        self.logger.critical(message)

if __name__ =='__main__':
    log_test = Logger('test.log',logging.ERROR,logging.DEBUG,True)
    log_test.error('Hello, world!!!')