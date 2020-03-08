---
title: 从零开始C++深度学习
author: 零露
tags:
  - 机器学习
  - C++
categories: 机器学习
mathjax: true
abbrlink: 971572897
date: 2020-03-03 18:41:38
---

## 前言

这篇文章原是我上学期的线代课的作业，后来添了一点东西又成了微积分作业，再后来又添了亿点东西又成了计导作业。现在再修修补补又是一篇博客`o(*￣ ▽ ￣*)ブ`。

## 框架

本文我们将只使用 STL 来实现一个最简单的手写数字识别（被誉为深度学习的 Hello world），之所以用简单的例子是因为难的我不会`(～￣ ▽ ￣)～`，或者也可以说是因为简单的例子比较能够看到问题的本质。

那么进入正题，我们首先列出一个框架，然后逢山开路，遇水造桥。我们知道深度学习涉及大量矩阵运算，因此我们首先需要实现一个矩阵类以表示矩阵及其运算。我们知道 python 中的线性代数库 numpy 的矩阵运算底层就是用 c 实现。一种实现是使用模板的，另一种是不使用模板的。两种实现各有利弊，如果使用模板将更符合数学的直觉——矩阵应该是不能随意改变形状的，并且许多行数和列数的匹配问题（比如矩阵乘法的行列匹配关系）可以在编译时发现；而不用模板的话可以更轻松地实现 reshape 的操作，并且可以更方便隐藏源代码。numpy 显然没有使用模板，而我当初写的时候没有考虑这么多，所以写成模板了`(´。＿。｀)`，然后就一直凑合用着。

然后，我们需要读取数据，我们知道深度学习是基于大数据的，没有数据就什么都没有。本文例子所用的数据来自于：http://yann.lecun.com/exdb/mnist/

如果做别的项目，互联网上其实也都有许多数据集可用使用，这就有待于聪明的你去发现了 `o(*￣ ▽ ￣*)ブ`。

有了数据集我们还需要读取，然后做一些处理，因此我们需要实现一个 Reader 类来读取数据。

之后，我们需要构建一个神经网络然后实现预测、训练等功能。

然后就是 main 用来调度。

那么本文以下的内容大概以此为主线展开：

- 实现 la::matrix
- 实现 Reader
- 实现 Framework
- 实现 main

我实现的矩阵类放在名称空间 `la`（linear algebra）中，此外还包含全局的操作符重载和应用于矩阵的函数，而其它几个类并没有塞进名称空间里面，可能稍微有点违和。

## 实现 la::matrix

我们知道 python 中的线性代数库 `numpy` 的矩阵运算底层就是用 C 实现。对于 C++的实现，我们有两种思路，一种实现是使用模板的，另一种是不使用模板的。两种实现各有利弊，如果使用模板将更符合数学的直觉——矩阵应该是不能随意改变形状的，并且许多行数和列数的匹配问题（比如矩阵乘法的行列匹配关系）可以在编译时发现；而不用模板的话可以更轻松地实现 `reshape` 的操作，并且可以更方便隐藏源代码。我所采用的是模板类的方式，在编码风格上模仿了`STL`。

为了方便管理，我们使用一维数组存储数据并提供 `at` 方法访问。为了应对不同的需求，诸如是否下标检查，是否 `const` 访问，提供四种 `at` 方法。为了防止堆栈空间不足而溢出，我们将其动态分配在堆上。
此外还可以提供迭代器以供快速迭代。
实现基本四则运算以及矩阵乘法，实现基本操作如转置，大多以操作符重载的形式实现。
此外还有一些杂乱的功能，例如适用于矩阵的函数，本例用到的其他矩阵操作等。

以下为全部代码。
```cpp
/******************
 * name: matrix.hpp
 * author: shizuku
*******************/
#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <cmath>

namespace la {
    template<size_t _M, size_t _N, typename _Ty = double>
    class matrix {
    public:
        matrix() {
            head = new _Ty[_M * _N]();
            tail = head + _M * _N;
            fill();
        }

        explicit matrix(const std::vector<_Ty> &a) {
            head = new _Ty[_M * _N]();
            tail = head + _M * _N;
            fill(a);
        }

        matrix(const matrix<_M, _N, _Ty> &src) {
            head = new _Ty[_M * _N]();
            tail = head + _M * _N;
            auto j = this->begin();
            for (auto i = src.cbegin(); i != src.cend(); ++i, ++j) {
                *j = *i;
            }
        }

        matrix(const matrix<_M, _N, _Ty> &&src) noexcept {
            head = new _Ty[_M * _N]();
            tail = head + _M * _N;
            auto j = this->begin();
            for (auto i = src.cbegin(); i != src.cend(); ++i, ++j) {
                *j = *i;
            }
        }

        ~matrix() { delete[] head; }

        class const_iterator {
        public:
            explicit const_iterator(_Ty *i) : item(i) {}

            const _Ty &operator*() const {
                return *item;
            }

            const _Ty *operator->() const {
                return item;
            }

            const _Ty &operator[](const size_t _Off) const {
                return *(item + _Off);
            }

            const_iterator &operator++() {
                ++item;
                return *this;
            }

            const_iterator operator++(int) {
                const_iterator _Tmp = *this;
                const_iterator::operator++();
                return _Tmp;
            }

            const_iterator &operator--() {
                --item;
                return *this;
            }

            const const_iterator operator--(int) {
                const_iterator _Tmp = *this;
                const_iterator::operator--();
                return _Tmp;
            }

            const_iterator &operator+=(const int i) {
                item += i;
                return *this;
            }

            const_iterator operator+(const int i) const {
                const_iterator _Tmp = *this;
                return _Tmp += i;
            }

            const_iterator &operator-=(const int i) {
                item -= i;
                return *this;
            }

            const_iterator operator-(const int _Off) const {
                const_iterator _Tmp = *this;
                return _Tmp -= _Off;
            }

            bool operator==(const const_iterator &_Right) const {
                return item == _Right.item;
            }

            bool operator!=(const const_iterator &_Right) const {
                return !(*this == _Right);
            }

            bool operator<(const const_iterator &_Right) const {

                return item < _Right.item;
            }

            bool operator>(const const_iterator &_Right) const {
                return _Right < *this;
            }

            bool operator<=(const const_iterator &_Right) const {
                return !(_Right < *this);
            }

            bool operator>=(const const_iterator &_Right) const {
                return !(*this < _Right);
            }

        private:
            _Ty *item;
        };

        class iterator {
        public:
            explicit iterator(_Ty *i) : item(i) {}

            _Ty &operator*() {
                return *item;
            }

            _Ty *operator->() {
                return item;
            }

            _Ty &operator[](const size_t _Off) {
                return *(item + _Off);
            }

            iterator &operator++() {
                ++item;
                return *this;
            }

            iterator operator++(int) {
                iterator _Tmp = *this;
                iterator::operator++();
                return _Tmp;
            }

            iterator &operator--() {
                --item;
                return *this;
            }

            iterator operator--(int) {
                iterator _Tmp = *this;
                iterator::operator--();
                return _Tmp;
            }

            iterator &operator+=(const int i) {
                item += i;
                return *this;
            }

            iterator operator+(const int i) const {
                iterator _Tmp = *this;
                return _Tmp += i;
            }

            iterator &operator-=(const int i) {
                item -= i;
                return *this;
            }

            iterator operator-(const int _Off) const {
                iterator _Tmp = *this;
                return _Tmp -= _Off;
            }

            bool operator==(const iterator &_Right) const {
                return item == _Right.item;
            }

            bool operator!=(const iterator &_Right) const {
                return !(*this == _Right);
            }

            bool operator<(const iterator &_Right) const {

                return item < _Right.item;
            }

            bool operator>(const iterator &_Right) const {
                return _Right < *this;
            }

            bool operator<=(const iterator &_Right) const {
                return !(_Right < *this);
            }

            bool operator>=(const iterator &_Right) const {
                return !(*this < _Right);
            }

        private:
            _Ty *item;
        };

        inline void clear() {
            fill();
        }

        void fill() {
            for (int i = 0; i < _M; i++) {
                for (int j = 0; j < _N; j++) {
                    at(i, j) = _Ty();
                }
            }
        }

        void fill(const std::vector<_Ty> &a) {
            int size = a.size();
            int k = 0;
            for (int i = 0; i < _M; i++) {
                for (int j = 0; j < _N; j++) {
                    if (k < size) {
                        fat(i, j) = a[k];
                        k++;
                    } else {
                        fat(i, j) = 0;
                        k++;
                    }
                }
            }
        }

        _Ty &at(size_t m, size_t n) {
            _Ty *r = (head + (m * _N + n));
            if (r < tail && r >= head) {
                return *r;
            } else {
                throw std::out_of_range("matrix out of range");
            }
        }

        _Ty &fat(size_t m, size_t n) {
            return *(head + (m * _N + n));
        }

        const _Ty &cat(size_t m, size_t n) const {
            _Ty *r = (head + (m * _N + n));
            if (r < tail && r >= head) {
                return *r;
            } else {
                throw std::out_of_range("matrix out of range");
            }
        }

        const _Ty &fcat(size_t m, size_t n) const {
            return *(head + (m * _N + n));
        }

        _Ty sum() {
            _Ty r = _Ty();
            for (int i = 0; i < _M; ++i) {
                for (int j = 0; j < _N; ++j) {
                    r += fat(i, j);
                }
            }
            return r;
        }

        size_t max_index() {
            size_t r = 0;
            size_t j = 1;
            _Ty k = *head;
            for (auto i = cbegin() + 1; i != cend(); ++i, ++j) {
                if (*i > k) {
                    k = *i;
                    r = j;
                }
            }
            return r;
        }

        _Ty maximum() {
            _Ty r = *head;
            for (auto i = cbegin() + 1; i != cend(); ++i) {
                if (*i > r) {
                    r = *i;
                }
            }
            return r;
        }

        inline const matrix<_N, _M, _Ty> transposition() const {
            return ~*this;
        }

        void save_as(std::string filename) {
            std::ofstream of(filename);
            for (int i = 0; i < _M; ++i) {
                for (int j = 0; j < _N; ++j) {
                    of << fat(i, j) << " ";
                }
            }
        }

        void read_from(std::string filename) {
            std::ifstream of(filename);
            for (int i = 0; i < _M; ++i) {
                for (int j = 0; j < _N; ++j) {
                    of >> fat(i, j);
                }
            }
        }

        matrix<_M, _N, _Ty> &operator=(const matrix<_M, _N, _Ty> &src) {
            if (this == &src) {
                return *this;
            }
            for (int i = 0; i < _M; i++) {
                for (int j = 0; j < _N; j++) {
                    fat(i, j) = src.fcat(i, j);
                }
            }
            return *this;
        }

        const matrix<_M, _N, _Ty> operator+() const {
            return *this;
        }

        const matrix<_M, _N, _Ty> operator-() const {
            matrix<_M, _N, _Ty> tmp(*this);
            for (auto i = tmp.begin(); i != tmp.end(); ++i) {
                *i = -*i;
            }
            return tmp;
        }

        const matrix<_N, _M, _Ty> operator~() const {
            matrix<_N, _M, _Ty> s = matrix<_N, _M, _Ty>();
            for (int i = 0; i < _N; i++) {
                for (int j = 0; j < _M; j++) {
                    s.fat(i, j) = this->fcat(j, i);
                }
            }
            return s;
        }

        matrix<_M, _N, _Ty> &operator+=(const matrix<_M, _N, _Ty> src) {
            return *this = *this + src;
        }

        matrix<_M, _N, _Ty> &operator-=(const matrix<_M, _N, _Ty> src) {
            return *this = *this - src;
        }

        template<class _TTy>
        matrix<_M, _N, _Ty> &operator*=(const _TTy src) {
            return *this = *this * _Ty(src);
        }

        template<class _TTy>
        matrix<_M, _N, _Ty> &operator/=(const _TTy src) {
            return *this = *this / _Ty(src);
        }

        template<size_t _A, typename _T = double>
        matrix<_A, _A, _Ty> &operator%=(const matrix<_A, _A, _Ty> src) {
            return *this = *this % src;
        }

        matrix::iterator begin() { return iterator(head); }

        matrix::iterator end() { return iterator(tail); }

        matrix::iterator rbegin() { return iterator(tail - 1); }

        matrix::iterator rend() { return iterator(head - 1); }

        const matrix::const_iterator cbegin() const { return const_iterator(head); }

        const matrix::const_iterator cend() const { return const_iterator(tail); }

        const matrix::const_iterator crbegin() const { return const_iterator(tail - 1); }

        const matrix::const_iterator crend() const { return const_iterator(head - 1); }

    private:
        _Ty *head;
        _Ty *tail;
    };

    template<size_t _A, size_t _B, typename _Ty>
    la::matrix<_A, _B, _Ty> operator+(const la::matrix<_A, _B, _Ty> &a, const la::matrix<_A, _B, _Ty> &b) {
        la::matrix<_A, _B, _Ty> s = la::matrix<_A, _B, _Ty>();
        for (int i = 0; i < _A; i++) {
            for (int j = 0; j < _B; j++) {
                s.fat(i, j) = a.fcat(i, j) + b.fcat(i, j);
            }
        }
        return s;
    }

    template<size_t _A, size_t _B, typename _Ty>
    la::matrix<_A, _B, _Ty> operator-(const la::matrix<_A, _B, _Ty> &a, const la::matrix<_A, _B, _Ty> &b) {
        la::matrix<_A, _B, _Ty> s = la::matrix<_A, _B, _Ty>();
        for (int i = 0; i < _A; i++) {
            for (int j = 0; j < _B; j++) {
                s.fat(i, j) = a.fcat(i, j) - b.fcat(i, j);
            }
        }
        return s;
    }

    template<size_t _A, size_t _B, typename _Ty, typename _T>
    la::matrix<_A, _B, _Ty> operator*(const la::matrix<_A, _B, _Ty> &a, const _T &b) {
        la::matrix<_A, _B, _Ty> s = la::matrix<_A, _B, _Ty>();
        for (int i = 0; i < _A; i++) {
            for (int j = 0; j < _B; j++) {
                s.fat(i, j) = a.fcat(i, j) * _Ty(b);
            }
        }
        return s;
    }

    template<size_t _A, size_t _B, typename _Ty, typename _T>
    inline la::matrix<_A, _B, _Ty> operator*(const _T &b, const la::matrix<_A, _B, _Ty> &a) {
        return operator*(a, b);
    }

    template<size_t _A, size_t _B, typename _Ty>
    la::matrix<_A, _B, _Ty> operator*(const la::matrix<_A, _B, _Ty> &a, const la::matrix<_A, _B, _Ty> &b) {
        la::matrix<_A, _B, _Ty> r{};
        for (int i = 0; i < _A; ++i) {
            for (int j = 0; j < _B; ++j) {
                r.fat(i, j) = a.fcat(i, j) * b.fcat(i, j);
            }
        }
        return r;
    }

    template<size_t _A, size_t _B, typename _Ty, typename _T>
    la::matrix<_A, _B, _Ty> operator/(const la::matrix<_A, _B, _Ty> &a, const _T &b) {
        la::matrix<_A, _B, _Ty> s = la::matrix<_A, _B, _Ty>();
        for (int i = 0; i < _A; i++) {
            for (int j = 0; j < _B; j++) {
                s.fat(i, j) = a.fcat(i, j) / _Ty(b);
            }
        }
        return s;
    }

    template<size_t _A, size_t _B, typename _Ty, typename _T>
    la::matrix<_A, _B, _Ty> operator/(const _T &b, const la::matrix<_A, _B, _Ty> &a) {
        la::matrix<_A, _B, _Ty> s = la::matrix<_A, _B, _Ty>();
        for (int i = 0; i < _A; i++) {
            for (int j = 0; j < _B; j++) {
                s.fat(i, j) = _Ty(b) / a.fcat(i, j);
            }
        }
        return s;
    }

    template<size_t _A, size_t _B, size_t _C, typename _Ty>
    la::matrix<_A, _C, _Ty> operator%(const la::matrix<_A, _B, _Ty> &a, const la::matrix<_B, _C, _Ty> &b) {
        la::matrix<_A, _C, _Ty> s = la::matrix<_A, _C, _Ty>();
        for (int i = 0; i < _A; i++) {
            for (int j = 0; j < _C; j++) {
                for (int k = 0; k < _B; k++) {
                    s.at(i, j) += a.cat(i, k) * b.cat(k, j);
                }
            }
        }
        return s;
    }

    template<size_t _A, size_t _B, typename _Ty>
    std::ostream &print(std::ostream &ostr, const la::matrix<_A, _B, _Ty> &a, int b) {
        int p1 = 0, p2 = 0, p3 = 0, p4 = 0;
        std::string x = "", y = "";
        if (b && _B > 10) {
            p1 = 5, p2 = _B - 5;
            x = "...\t";
        } else {
            p1 = _B / 2;
            p2 = p1;
        }
        if (b && _A > 10) {
            p3 = 5, p4 = _A - 5;
            y = " ...\n";
        } else {
            p3 = _A / 2;
            p4 = p3;
        }
        char h = '[';
        for (int i = 0; i < p3; i++) {
            ostr << h;
            if (i == 0) h = ' ';
            ostr << "[\t";
            for (int j = 0; j < p1; j++) {
                ostr << a.fcat(i, j) << "\t";
            }
            ostr << x;
            for (int j = p2; j < _B; j++) {
                ostr << a.fcat(i, j);
                if (j != _B - 1)
                    ostr << "\t";
            }
            ostr << "\t]\n";
        }
        ostr << y;
        for (int i = p4; i < _A; i++) {
            ostr << h << "[\t";
            for (int j = 0; j < p1; j++) {
                ostr << a.fcat(i, j) << "\t";
            }
            ostr << x;
            for (int j = p2; j < _B; j++) {
                ostr << a.fcat(i, j);
                if (j != _B - 1)
                    ostr << "\t";
            }
            ostr << "\t]";
            if (i != _A - 1)ostr << "\n";
        }
        ostr << "] " << _A << "*" << _B << "\n";
        return ostr;
    }

    template<size_t _A, size_t _B, typename _Ty>
    std::ostream &operator<<(std::ostream &ostr, const la::matrix<_A, _B, _Ty> &a) { return print(ostr, a, 0); }

    template<size_t _A, size_t _B, typename _Ty>
    const la::matrix<_A, _B, _Ty> transposition(const la::matrix<_B, _A, _Ty> &x) {
        return ~x;
    }

    template<size_t _A, typename _Ty>
    const la::matrix<_A, _A, _Ty> diag(const la::matrix<1, _A, _Ty> &src) {
        la::matrix<_A, _A, _Ty> r{};
        for (int i = 0; i < _A; i++) {
            r.fat(i, i) = src.fcat(0, i);
        }
        return r;
    }

    template<size_t _A, size_t _B, typename _Ty>
    const la::matrix<_A, _B, _Ty> outer(const la::matrix<1, _A, _Ty> &a, const la::matrix<1, _B, _Ty> &b) {
        la::matrix<_A, _B, _Ty> r{};
        for (int i = 0; i < _A; ++i) {
            for (int j = 0; j < _B; ++j) {
                r.fat(i, j) = a.fcat(0, i) * b.fcat(0, j);
            }
        }
        return r;
    }

    template<size_t _A, size_t _B, size_t _C, size_t _D, typename _Ty>
    const la::matrix<_A, _B, _Ty> reshape(const la::matrix<_C, _D, _Ty> &x) {
        if (_A * _B == _C * _D) {
            la::matrix<_A, _B, _Ty> r = la::matrix<_A, _B, _Ty>();
            auto i = r.begin();
            auto j = x.cbegin();
            for (; i != r.end(); ++i, ++j) {
                *i = *j;
            }
            return r;
        } else {
            throw std::out_of_range(" ");
        }
    }

    template<size_t _M, size_t _N, class _T>
    la::matrix<_M, _N, _T> power(const la::matrix<_M, _N, _T> &x, const int y) {
        la::matrix<_M, _N, _T> r = la::matrix<_M, _N, _T>();
        for (int i = 0; i < _M; i++) {
            for (int j = 0; j < _N; j++) {
                r.fat(i, j) = std::pow(x.fcat(i, j), y);
            }
        }
        return r;
    }

    template<size_t _M, class _T>
    la::matrix<1, _M, _T> softmax(const la::matrix<1, _M, _T> &x) {
        la::matrix<1, _M, _T> x_exp = la::matrix<1, _M, _T>();
        la::matrix<1, _M, _T> r = la::matrix<1, _M, _T>();

        for (int i = 0; i < _M; ++i) {
            x_exp.fat(0, i) = exp(x.fcat(0, i));
        }
        _T sum_x_exp = x_exp.sum();
        for (int i = 0; i < _M; ++i) {
            r.fat(0, i) = x_exp.fcat(0, i) / sum_x_exp;
        }
        return r;
    }

    template<size_t _M, class _T>
    la::matrix<_M, _M, _T> dsoftmax(const la::matrix<1, _M, _T> &x) {
        auto sm = softmax(x);
        return la::diag(sm) - la::outer(sm, sm);
    }

    template<size_t _M, size_t _N, class _T>
    la::matrix<_M, _N, _T> cosh(const la::matrix<_M, _N, _T> &x) {
        la::matrix<_M, _N, _T> r = la::matrix<_M, _N, _T>();
        for (int i = 0; i < _M; i++) {
            for (int j = 0; j < _N; j++) {
                r.fat(i, j) = std::cosh(x.fcat(i, j));
            }
        }
        return r;
    }

    template<size_t _M, size_t _N, class _T>
    la::matrix<_M, _N, _T> tanh(const la::matrix<_M, _N, _T> &x) {
        la::matrix<_M, _N, _T> r = la::matrix<_M, _N, _T>();
        for (int i = 0; i < _M; i++) {
            for (int j = 0; j < _N; j++) {
                r.fat(i, j) = std::tanh(x.fcat(i, j));
            }
        }
        return r;
    }

};

```

## 实现 Reader

根据`mnist`页面的描述，我们编写一个类来读取此数据集。

根据描述，`img`数据的开头为四个 32 位`int`，我们将其读取并忽略。而`lab`数据开头为两个 32 位`int`，同样读取并忽略。如构造函数所示。

`img` 数据为四位表示一个像素点，我们需要连续读取四个位并将其反转，读取这样的 784 个数字并且将其存储在矩阵中，然后返回。`lab` 数据只需要读取一个数字并返回即可。

在`get_img`和`get_lab`函数中用`read`方法读取，`img`数据需要反转，所以添加`private`方法`reverse_32`。

简略代码如下：

```cpp
template<size_t _M, size_t _N>
class Reader {
public:
    Reader(std::string img, std::string lab) {...}
    const la::matrix<_M, _N> get_img(const size_t index) {...}
    const char get_lab(const int index) {...}
private:
    unsigned int reverse_32(unsigned int n) {...}
    std::ifstream img;
    std::ifstream lab;
};
```

## 实现 Framework

接下来进行神经网络的实现，我们的神经网络非常简单，只有一个隐藏层，隐藏层也只有十个神经元。对此我们编写一个类实现。类包含 \(b0\)、\(b1\)、\(w1\)三个矩阵。此外还实现了预测`predict`和训练`train`的功能。

在构造函数中我们随机初始化\(w1\)矩阵，随机区间为\([-\sqrt{\frac{6}{28 \times 28 + 10}},\sqrt{\frac{6}{28 \times 28 + 10}}]\)，\(b0\)、\(b1\)则全部为 \(0\)。

预测只需要根据式(5)进行即可。

```cpp
la::matrix<1, 10> NeuralNetwork::predict(const la::matrix<1, 28 * 28>& in) {
    return f2((((f1((in + b0))) % w1) + b1));
}
```

训练需要以`train_batch`为单位进行，我们以 100 个数据为一组对神经网络进行调整，总共 600 组。

```cpp
void NeuralNetwork::train(Reader<size1, size1>& data) {
    la::matrix<1, size1 * size1> b0tmp{};
    la::matrix<1, size2> b1tmp{};
    la::matrix<size1 * size1, 10> w1tmp{};
    int lab{};
    la::matrix<1, size1 * size1> in{};
    la::matrix<1, size1 * size1> l0_in{};
    la::matrix<1, size1 * size1> l0_out{};
    la::matrix<1, size2> l1_in{};
    la::matrix<1, size2> l1_out{};
    la::matrix<size2, 1> act1{};
    la::matrix<size2, 1> grad_b1{};
    la::matrix<size1 * size1, 10> grad_w1{};
    la::matrix<1, size1 * size1> grad_b0{};
    for (int i = 0; i < 600; i++) {
        for (int j = 0; j < 100; j++) {
            lab = data.get_lab(0);
            in = la::reshape<1, size1 * size1>(data.get_img(0));
            l0_in = in + b0;
            l0_out = f1(l0_in);
            l1_in = ((l0_out % w1) + b1);
            l1_out = f2(l1_in);
            act1 = (df2(l1_in) % ~(la::matrix<1, size2>(identity(lab)) - l1_out));
            grad_b1 = -2 * act1;
            grad_w1 = -2 * la::outer(l0_out, ~act1);
            grad_b0 = -2 * (df1(l0_in) * ~(w1 % act1));
            b1tmp += ~grad_b1;
            w1tmp += grad_w1;
            b0tmp += grad_b0;
        }
        b0 -= (b0tmp / 100) * learn_rate;
        w1 -= (w1tmp / 100) * learn_rate;
        b1 -= (b1tmp / 100) * learn_rate;
        b0tmp.clear();
        w1tmp.clear();
        b1tmp.clear();
        std::cout << i << "\n";
    }
}
```

以下为类的声明：

```cpp
const size_t size1 = 28;
const size_t size2 = 10;
class NeuralNetwork {
public:
    NeuralNetwork();
    la::matrix<1,10> predict(const la::matrix<1, 28 * 28>& in);
    void train(Reader<size1, size1>& data);
    void train(const std::string& img, const std::string& lab);
    void save();
    void read();
private:
    static const std::vector<double> identity(int a, int b = 10);
    double learn_rate;
    la::matrix<1, size1 * size1> b0;
    la::matrix<1, size2> b1;
    la::matrix<size1 * size1, 10> w1;
};
```

## 实现 main

`main` 模块主要包含 `main` 函数调度和正确率计算功能。

我们在 `main` 函数中实例化神经网络，并进行训练，分别输出训练前和训练后的证确率。代码如下：

```cpp
int main() {
    NeuralNetwork dm = NeuralNetwork();

    std::cout << test(dm, test_img_filename, test_lab_filename) << "\n";
    system("pause");

    dm.train(train_img_filename, train_lab_filename);

    std::cout << test(dm, test_img_filename, test_lab_filename) << "\n";
    system("pause");
}
```

正确率计算并不困难，我们只需要对比期望的结果和实际的结果，并计数即可，代码如下：

```cpp
double test(NeuralNetwork& dm, const std::string& img, const std::string& lab){
    Reader<size1, size1> test(test_img_filename, test_lab_filename);
    int a = 0, b = 0;
    std::vector<int> num1(10,0);
    std::vector<int> num2(10,0);
    for (int i = 0; i < test_num; ++i) {
        auto img_i = test.get_img(0);
        int lab_i = (int)test.get_lab(0);
        auto rimg = dm.predict(la::reshape<1, 28 * 28>(img_i));
        int m = (int)rimg.max_index();
        if (m == lab_i) {
            a++; b++;
        }
        else {
            b++;
        }
        num1[m]++;
        num2[lab_i]++;
    }
    return double(a) / double(b);
}
```
![运行结果]( https://cdn.jsdelivr.net/gh/shizuku/shizuku.github.io@master/posts/971572897/result.png )

如图为实验的结果，最终的正确率大概有92%，虽然由于神经网络的结构过于简单，致使训练后的正确率并不高，但这样的结果也符合我们的预期。
