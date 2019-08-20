#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def add01():
    a = np.ones((2, 2))
    b = np.arange(4).reshape(2, 2)
    c = a + b
    print(c)
    print(c.shape)


def add02():
    a = np.ones((2, 2))
    b = np.arange(2).reshape(2)  # ⇒ ルール1より、(1, 2)になる
    c = a + b
    print(c)
    print(c.shape)


def add03():
    a = np.ones((2, 2))
    b = np.arange(2).reshape(2, 1)
    c = a + b
    print(c)
    print(c.shape)


def add04():
    a = np.ones((2, 2))  # ⇒ ルール1より、(1, 2, 2)になる
    b = np.arange(2).reshape(2, 1, 1)
    c = a + b
    print(c)
    print(c.shape)


def add05():
    a = np.ones((2, 2))
    b = np.arange(2).reshape(2, 1, 1, 1)
    c = a + b
    print(c)
    print(c.shape)


def add06():
    a = np.ones((5, 6))
    b = np.arange(5).reshape(5, 1)
    c = a + b
    print(c)
    print(c.shape)


if __name__ == "__main__":
    """
    ブロードキャスティングのルール

    1. 最も次元数の多い行列よりも少ない次元数を持つ行列は先頭に行が追加される。
    例：(5, 6)と(5,)　⇒ (5, 6)と(1, 5)の様に解釈される。

    2. 出力される行列のそれぞれの次元サイズは、すべての入力サイズのmaxとなる。
    例：(5, 6)と(5,) ⇒ (5, 6)と(1, 5) ⇒ (5, 6)

    3. 特定の次元のサイズが、出力サイズの次元に一致する場合、もしくはサイズ１の次元を
    持つ場合、ブロードキャストが行われる。

    4. もしも入力行列にサイズ1の次元が存在するとき、対応する次元全てにその値が作用する。

    例：(5, 6) (6, 5)これは計算できない
    例：(5, 6) (5, )これは計算できない
    例：(5, 6) (6, ) ⇒ (5, 6) (1, 6) ⇒ (5, 6)これは計算できる

    """
    add01()
    add02()
    add03()
    add04()
    add05()
    add06()

    a = np.sum(np.array([1, 1]))
    print(a.reshape(1, 1))
