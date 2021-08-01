#-------------------------------------------------
#
# Project created by QtCreator 2017-09-04T16:14:43
#
#-------------------------------------------------

QT       += core
QT       -= gui

TARGET = _nn00
CONFIG   += console
CONFIG   -= app_bundle
CONFIG   += c++11

TEMPLATE = app


QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS   += -fopenmp

QMAKE_CXXFLAGS_RELEASE += -O3
QMAKE_CXXFLAGS_RELEASE -= -O2
QMAKE_CXXFLAGS_RELEASE -= -O1

QMAKE_LFLAGS_RELEASE   += -O3
QMAKE_LFLAGS_RELEASE   -= -O2
QMAKE_LFLAGS_RELEASE   -= -O1


SOURCES += \
    src/main.cpp \
    src/misc/benchmark.cpp \
    src/misc/args.cpp \
    src/misc/tools.cpp \
    src/misc/rand.cpp \
    src/misc/critical.cpp \
    src/engine/nnnet.cpp \
    src/engine/data.cpp

HEADERS += \
    src/misc/benchmark.h \
    src/misc/rand.h \
    src/misc/critical.h \
    src/misc/args.h \
    src/misc/tools.h \
    src/engine/nnnet.h \
    src/engine/data.h \
    src/defs.h

OTHER_FILES += \
    nndef.txt \
    nndef_out.txt \
    data.txt

DISTFILES += \
    tmp


