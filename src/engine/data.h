#pragma once

#include <QVector>
#include <QString>

#include "src/defs.h"
#include "src/misc/critical.h"
#include "src/misc/rand.h"

namespace NsNet {


class NNRecord {
private:
    QVector<QString> desc;
    TVFlt    inp;
    TVFlt    out;

public:
    NNRecord() {}
    NNRecord(const QVector<QString> &desc, CTVFlt &inp, const CTVFlt &out) :
        desc(desc), inp(inp), out(out) {
    }
    const QString& getDesc(nncityp nr) const { return desc[nr]; }
    nnftyp getInp (nncityp nr) const { return inp[nr];  }
    nnftyp getOut (nncityp nr) const { return out[nr];  }

    const QVector<QString>& getDescs() const { return desc; }
    CTVFlt& getInps() const { return inp; }
    CTVFlt& getOuts() const { return out; }

    nnityp sizeDesc() const { return desc.size(); }
    nnityp sizeInp() const { return inp.size(); }
    nnityp sizeOut() const { return out.size(); }
    nnityp size() const { return sizeDesc() + sizeInp() + sizeOut(); }

    QString operator[] (nnityp nr) const {
        if( nr < desc.size() ) return desc[nr];
        nr -= desc.size();
        if( nr < inp.size()  ) return QString::number( (double)inp[nr] );
        nr -= inp.size();
        if( nr < out.size()  ) return QString::number( (double)out[nr] );
        EXCP( false );
        return "";
    }

    void setDesc( nncityp nr , const QString &v ) {
        EXCP( nr < desc.size() );
        desc[nr] = v;
    }
    void setInp( nncityp nr , nncftyp v ) {
        EXCP( nr < inp.size() );
        inp[nr] = v;
    }
    void setOut( nncityp nr , nncftyp v ) {
        EXCP( nr < out.size() );
        out[nr] = v;
    }

    void appendDesc( const QString &v ) {
        desc.append(v);
    }
    void appendInp( nncftyp v ) {
        inp.append(v);
    }
    void appendOut( nncftyp v ) {
        out.append(v);
    }
};

typedef const NNRecord CNNRecord;


class NNData : public QVector<NNRecord> {
public:
    QVector<QString> heads;

    nnityp cols() const { return first().size(); }

    nnityp sizeDesc() const { return first().sizeDesc(); }
    nnityp sizeInp() const { return first().sizeInp();   }
    nnityp sizeOut() const { return first().sizeOut();   }

    void append( CNNRecord &r ) {
        EXCP( size() == 0 || r.sizeDesc() == sizeDesc() );
        EXCP( size() == 0 || r.sizeInp() == sizeInp() );
        EXCP( size() == 0 || r.sizeOut() == sizeOut() );
//        for( nnityp i=0 ; i<r.size() ; i++ )
//            printf("%s ",qPrintable(r[i]) );
//        printf("\n");
//        fflush(stdout);
        QVector<NNRecord>::append( r );
    }

    void append( const QVector<QString> &desc, CTVFlt &inp, CTVFlt &out ) {
        const NNRecord r(desc,inp,out);
        append(r);
    }

    void setHeads( const QVector<QString> &heads ) {
        this->heads = heads;
    }

    const QString getHead( nncityp nr ) const {
        return nr < heads.size() ? heads[nr] : QString::number(nr+1);
    }

    static NNData mkData(
        const bool heads,      // czy kolumny mają nagłówki?
        nncityp size_desc,      // ilość pól opisowych
        nncityp size_inp,       // ilość wejść
        nncityp begin_inp,      // początkowe wejście
        nncityp end_inp,        // końcowe wejście
        nncityp size_out,       // ilość wyjść
        const QString &path,   // ścieżka do pliku
        const QString &sep     // separator pola
    );


    NNData rndSelect(
        nnityp  size1,
        nncityp rndSeed
    ) const;

    NNData nSelect(
        nnityp size1
    ) const;

    void split(
        NNData &out1,
        nnityp size1,
        NNData &out2,
        nnityp size2,
        nncityp rndSeed
   ) const;

    void split(
        NNData &out1,
        nnityp size1,
        NNData &out2,
        nnityp size2,
        NNData &out3,
        nnityp size3,
        nncityp rndSeed
    ) const;

    void shuffle( nncityp rndSeed );
    void shuffle( nncityp rndSeed , nncityp size );

};

typedef const NNData CNNData;

}
