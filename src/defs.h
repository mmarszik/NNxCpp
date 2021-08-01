#pragma once

#include <QVector>
#include <QTextStream>

static inline QTextStream& operator << (QTextStream& ts, long double x) {
    return ts << (double)x;
}

static inline QTextStream& operator << (QTextStream& ts, __float128 x) {
    return ts << (double)x;
}

typedef double ftyp;
typedef const ftyp cftyp;

typedef long double lftyp;
typedef const lftyp clftyp;

typedef int ityp;
typedef const ityp cityp;

typedef unsigned int utyp;
typedef const utyp cutyp;

typedef long long ltyp;
typedef const long long cltyp;

typedef unsigned long long ultyp;
typedef const unsigned long long cultyp;

namespace NsNet {

typedef double nnftyp;          // Typ zmienno-przecinkowy dla sieci neuronowej
typedef const nnftyp nncftyp;   // Typ stały zmienno-przecinkowy dla sieci neuronowej

typedef int nnityp;
typedef const nnityp nncityp;

#define NNVec QVector

// Wektor intów
typedef NNVec<int>  TVInt;
typedef const TVInt CTVInt;
typedef NNVec<TVInt> TVIInt;

// Wektor floatów
typedef NNVec<nnftyp> TVFlt;
typedef const TVFlt   CTVFlt;

typedef NNVec<QString> TVStr;
typedef const TVStr CTVStr;


}
