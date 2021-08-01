#include "data.h"

#include <QFile>
#include <QString>
#include <QTextStream>
#include <QStringList>

namespace NsNet {

NNData NNData::mkData(
    const bool heads,     // czy kolumny mają nagłówki?
    nncityp size_desc,      // ilość pól opisowych
    nncityp size_inp,       // ilość wejść
    nncityp begin_inp,      // początkowe wejście
    nncityp end_inp,        // końcowe wejście
    nncityp size_out,       // ilość wyjść
    const QString &path , // ścieżka do pliku
    const QString &sep    // separator pola
) {
    bool ok;
    NNData data;

    QFile file(path);
    PEXCP( file.open( QFile::ReadOnly ) );

    nncityp size = size_desc + size_inp + size_out;
    QTextStream stream( &file );
    QRegExp rx(sep);

    QString line;
    for( nnityp row=0 ; ! (line = stream.readLine()).isNull() ; row++ ) {
        line = line.simplified();
        const QStringList list = line.split( rx , QString::KeepEmptyParts );

//        printf("%d (%d)", (int)row , (int)list.size() );
//        for( nnityp i=0 ; i<list.size() ; i++ ) {
//            printf("%s" , qPrintable(list[i]) );
//        }
//        printf("\n");
//        fflush(stdout);

        if( list.size() != size ) {
            printf("%s" , qPrintable( QString("list.size() != size\n%1 != %2\n").arg(list.size()).arg(size) ) );
            fflush(stdout);
        }
        PEXCP( list.size() == size );

        if( row==0 && heads ) {
            data.setHeads( list.toVector() );
        } else {
            NNRecord r;
            for( nnityp i=0 ; i<size ; i++ ) {
//                printf("i=%d\n",i);
//                fflush(stdout);
                if( i < size_desc ) {
                    r.appendDesc( list[i] );
                } else if( i < size_desc + size_inp ) {
                    if( i-size_desc >= begin_inp && i-size_desc <= end_inp ) {
                        r.appendInp( QString(list[i]).replace( "," , "." ).toDouble(&ok) );
                        EXCP( ok );
                    }
                } else {
                    r.appendOut( QString(list[i]).replace( "," , "." ).toDouble(&ok) );  EXCP( ok );
                }
            }
            data.append( r );
        }
    }

    return data;
}



NNData NNData::rndSelect(
    nnityp size1,
    FRnd &frnd
) const {
    QVector<nnityp> idx( size() );
    for( nnityp i=0 ; i<size() ; i++ )
        idx[i] = i;
    for( nnityp i=0 ; i<size()-1 ; i++ )
        qSwap( idx[ i ] , idx[ i + frnd.getI() % (size()-i) ] );
    NNData out;
    for( nnityp i=0 ; i<size() && i<size1 ; i++ )
        out.append( at( idx[ i ] ) );
    return out;
}

NNData NNData::nSelect(
    nnityp size1
) const {
    NNData out;
    for( nnityp i=0 ; i<size() && i<size1 ; i++ ) {
        out.append( at( i ) );
    }
    return out;
}


void NNData::split(
    NNData &out1,
    nnityp size1,
    NNData &out2,
    nnityp size2,
    FRnd   &frnd
) const {
    QVector<nnityp> idx( size() );

    for( nnityp i=0 ; i<size() ; i++ ) {
        idx[i] = i;
    }

    for( nnityp i=0 ; i<size()-1 ; i++ ) {
        qSwap(idx[ i ], idx[ i + frnd.getI() % (size()-i) ]);
    }

    out1.clear();
    out2.clear();

    for( nnityp i=0 ; i<size() && i<size1 ; i++ ) {
        out1.append( at( idx[ i ] ) );
    }

    for( nnityp i=size1 ; i<size() && i<size1+size2 ; i++ ) {
        out2.append( at( idx[ i ] ) );
    }

}




//void Data::split(
//    Data &out1,
//    nnityp size1,
//    Data &out2,
//    nnityp size2,
//    Data &out3,
//    nnityp size3
//) const {
//    QVector<nnityp> idx( size() );
//    for( nnityp i=0 ; i<size() ; i++ )
//        idx[i] = i;
//    for( nnityp i=0 ; i<size()-1 ; i++ ) {
//        idx[i] = idx[ i + irand() % (size()-i) ];
//    }
//    out1.clear();
//    out2.clear();
//    out3.clear();
//    for( nnityp i=0 ; i<size() && i<size1 ; i++ )
//        out1.append( at(i) );
//    for( nnityp i=size1 ; i<size() && i<size1+size2 ; i++ )
//        out2.append( at(i) );
//    for( nnityp i=size1+size2 ; i<size() && i<size1+size2+size3 ; i++ )
//        out3.append( at(i) );
//}


void NNData::split(
    NNData &out1,
    nnityp size1,
    NNData &out2,
    nnityp size2,
    NNData &out3,
    nnityp size3,
    FRnd &frnd
) const {
    QVector<nnityp> idx( size() );

    for( nnityp i=0 ; i<size() ; i++ )
        idx[i] = i;

    for( nnityp i=0 ; i<size()-1 ; i++ ) {
        qSwap(idx[ i ] , idx[ i + frnd.getI() % (size()-i) ] );
    }

    out1.clear();
    out2.clear();
    out3.clear();

    for( nnityp i=0           ; i<size() && i<size1             ; i++ )
        out1.append( at( idx[ i ] ) );
    for( nnityp i=size1       ; i<size() && i<size1+size2       ; i++ )
        out2.append( at( idx[ i ] ) );
    for( nnityp i=size1+size2 ; i<size() && i<size1+size2+size3 ; i++ )
        out3.append( at( idx[ i ] ) );

}

void NNData::shuffle( FRnd &rnd ) {
    for( nnityp i=0 ; i<size()-1 ; i++ ) {
        qSwap( (*this)[ i ] , (*this)[ i + rnd.getI() % (size()-i) ] );
    }
}

void NNData::shuffle(FRnd &rnd, nncityp size) {
    if( this->size() > size ) {
        abort();
    }
    for( nnityp i=0 ; i<size-1 ; i++ ) {
        qSwap( (*this)[ i ] , (*this)[ i + rnd.getI() % (size-i) ] );
    }
}

}


/*
bool Data::read(
    const QString &path , // ścieżka do pliku
    const QString &sf ,   // separator pola
    const QString &sd     // separator dziesiętny
) {
    QFile file(path);
    if( ! file.open( QFile::ReadOnly) )
        return false;
    QVector<QVector<ftyp>> tmp;
    QTextStream stream( &file );

    while( true ) {
        const QString line = stream.readLine().simplified();
        if( line.size() < 1 )
            break;
        const QStringList list = line.split( sf );
        if( tmp.size() != 0 && tmp.first().size() != list.size() )
            return false;
        QVector<ftyp> v;
        for( nnityp i=0 ; i<list.size() ; i++ ) {
            bool ok;
            v.append( list[i].toDouble(&ok) );
            if( !ok )
                return false;
        }
        v.squeeze();
        tmp.append( v );
    }
    tmp.squeeze();
    data = tmp;
    return true;
}
*/
