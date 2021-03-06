#include <cassert>
#include <limits>
#include <omp.h>
#include <QFile>
#include <QTextStream>
#include <QRegExp>
#include <cmath>
#include <QtDebug>
#include "nnnet.h"
#include "src/misc/critical.h"

namespace NsNet {

template<typename T>
T signum(const T value) {
    return value < static_cast<T>(0) ? -1 : + 1;
}

static nnftyp factivateMLin( nncftyp inp ) {
    nnftyp a,b;
    if     ( inp < -10.0 ) { a=0.01; b=-1.7; }
    else if( inp < - 3.0 ) { a=0.10; b=-0.8; }
    else if( inp < - 1.0 ) { a=0.30; b=-0.2; }
    else if( inp < + 1.0 ) { a=0.50; b= 0.0; }
    else if( inp < + 3.0 ) { a=0.30; b=+0.2; }
    else if( inp < +10.0 ) { a=0.10; b=+0.8; }
    else                   { a=0.01; b=+1.7; }
    return a * inp + b;
}

static nnftyp fderivateMLin( nncftyp out ) {
    if( out > +1.8 ) return 0.01;
    if( out > +1.1 ) return 0.10;
    if( out > +0.5 ) return 0.30;
    if( out > -0.5 ) return 0.50;
    if( out > -1.1 ) return 0.30;
    if( out > -1.8 ) return 0.10;
    return 0.01;
}


// Funkcja aktywacji
static nnftyp factivate( nncftyp inp, NNActv actv ) {
    switch( actv ) {
        case NNA_LIN:   return inp;
        case NNA_MLIN:  return factivateMLin(inp);
        case NNA_SUNI:  return 1.0 / ( 1.0 + exp(-inp) );
        case NNA_SBIP:  return 2.0 / ( 1.0 + exp(-inp) ) - 1.0;
        case NNA_SBIP_L:return 2.0 / ( 1.0 + exp(-inp) ) - 1.0 + inp * 1E-6;
        case NNA_RELU:  return log( 1.0 + exp(inp) );
        case NNA_NNAME: return inp / ( 1.0 + fabs(inp) );
    }
    return 0;
}

// Funkcja pochodna
static nnftyp fderivate( nncftyp out, NNActv actv ) {
    nnftyp tmp;
    switch( actv ) {
        case NNA_LIN:    return 1;
        case NNA_MLIN:   return fderivateMLin(out);
        case NNA_SUNI:   return out * (1.0 - out);
        case NNA_SBIP:   return 0.5 * (1.0 - out*out);
        case NNA_SBIP_L: return 0.5 * (1.0 - out*out) + 1E-6;
        case NNA_RELU:   return 1.0 / ( exp(-out) + 1 );
        case NNA_NNAME:  tmp = out < 0 ? out/(1.0+out) : -(out/(out-1.0)); tmp = fabs(tmp)+1.0; tmp *= tmp; return 1.0 / tmp;
    }
    return 0;
}

/*
void NNNet::compute( CTVFlt &inp , TVFlt &out , TVFlt &obuf ) const {
    EXCP( inp.size() == size_i );         // Sie?? jest przystosowana do jednego rozmiaru wej??cia i
    EXCP( out.size() == size_o );         // wyj??cia.

    for( nnityp i=0 ; i<size_i ; i++ ) {  // Skopiuj wej??cia.
        obuf[i] = inp[i];
    }

    // obliczenia
    for( nnityp i=0 ; i<neurons.size() ; i++ ) {  // Po wszystkich neuronach.
        CNNNeuron &neuron = neurons[i];           // Skr??t do bie????cego neuronu.
        nnftyp nout = 0;                          // Pami???? na wyj??cie neuronu inicjuj zerem.
        CTVInt &idx_i = neuron.idx_i;             // Skr??t do indeks??w wej????.
        CTVInt &idx_w = neuron.idx_w;             // Skr??t do indeks??w wag.
        EXCP( idx_i.size() == idx_w.size() );     // Ilo???? wag i wej???? musi by?? taka sama.
        for( nnityp j=0 ; j<idx_i.size() ; j++ ) {
            if( idx_i[j] >= i + size_i ) abort();
            nout += weights[ idx_w[j] ] * obuf[ idx_i[j] ];  // Suma wa??ona.
        }
        obuf[ i + size_i ] = factivate( nout , neuron.actv ); // Wyj??cie neuronu, potem zoptymalizowa??.
    }

    for( nnityp i=0 ; i<size_o ; i++ ) {       // Po wszystkich wyj??ciach.
        out[i] = obuf[obuf.size()-size_o+i]; // Na ko??cu jest odpowied?? sieci, skopiuj do wyj??cia.
    }

}



void NNNet::compute( nncftyp *inp , nnftyp *obuf ) const {
    nncftyp *const end_no = obuf + size_i + neurons.size();
    nncftyp *const weights = this->weights.constData();

    {
        nncftyp *const end_inp = inp + size_i;
        while( inp < end_inp )
            *obuf++ = *inp++;
    }

    {
        CNNNeuron *neuron = neurons.constData();
        while( obuf < end_no ) {
            *obuf = 0;
            nncityp *idx_i = neuron->idx_i.constData();
            nncityp *const end_idx_i = idx_i + neuron->idx_i.size();
            nncityp *idx_w = neuron->idx_w.constData();
            while( idx_i < end_idx_i ) {
                *no += weights[ *idx_w++ ] * obuf[ *idx_i++ ];
            }
            *obuf = factivate( *obuf , neuron->actv );
            obuf++ ;
            neuron++;
        }
    }

}

void NNNet::compute(CTVFlt &inp , TVFlt &out ) const {
    TVFlt obuf( size_i + neurons.size() ); // Pami???? na wyj??cia neuron??w.
    compute( inp , out , obuf );
}

nnftyp NNNet::error( CTVFlt &inp , CTVFlt &out , TVFlt &obuf  ) const {
    nnftyp sum = 0;
    compute( inp.constData() , obuf.data() );
    for( nnityp i=0 ; i<size_o ; i++ ) {
        nncftyp t = obuf[size_i+neurons.size()-size_o+i] - out[i];
        sum += t * t;
    }
    return sum;
}

nnftyp NNNet::error( CTVFlt &inp , CTVFlt &out ) const {
    TVFlt obuf( size_i + neurons.size() ); // Pami???? na wyj??cia neuron??w.
    return error( inp , out , obuf );
}

nnftyp NNNet::error( CNNRecord &rec , TVFlt &obuf ) const {
    return error( rec.getInps() , rec.getOuts() , obuf );
}

nnftyp NNNet::error( CNNRecord &rec ) const {
    TVFlt obuf( size_i + neurons.size() ); // Pami???? na wyj??cia neuron??w.
    return error( rec , obuf );
}

nnftyp NNNet::error( CNNData &data ) const {
    nnftyp sum = 0;
    nncityp size = data.size();
    TVFlt obuf( size_i + neurons.size() );
//#pragma omp parallel for reduction(+:sum) firstprivate(obuf)
    for( nnityp i=0 ; i<size ; i++ ) {
        sum += error( data[i] );
    }
    return sqrt( sum / data.size() / size_o );
}
*/

void NNNet::compute( CTVFlt &inp , TVFlt &obuf ) const {
    EXCP( inp.size() == size_i );                     // Sie?? jest przystosowana do jednego rozmiaru wej??cia i
    EXCP( obuf.size() == size_o + neurons.size() );   // wyj??cia.
    for( nnityp i=0 ; i<size_i ; i++ ) {              // Skopiuj wej??cia.
        obuf[i] = inp[i];
    }
    // obliczenia
    for( nnityp i=0 ; i<neurons.size() ; i++ ) {  // Po wszystkich neuronach.
        CNNNeuron &neuron = neurons[i];           // Skr??t do bie????cego neuronu.
        nnftyp nout = 0;                          // Pami???? na wyj??cie neuronu inicjuj zerem.
        CTVInt &idx_i = neuron.idx_i;             // Skr??t do indeks??w wej????.
        CTVInt &idx_w = neuron.idx_w;             // Skr??t do indeks??w wag.
        EXCP( idx_i.size() == idx_w.size() );     // Ilo???? wag i wej???? musi by?? taka sama.
        for( nnityp j=0 ; j<idx_i.size() ; j++ ) {
//            qDebug() << idx_w[j] << weights[ idx_w[j] ];
            nout += weights[ idx_w[j] ] * obuf[ idx_i[j] ];  // Suma wa??ona.
        }
        obuf[ i + size_i ] = factivate( nout , neuron.actv ); // Wyj??cie neuronu, potem zoptymalizowa??.
    }
}


TVFlt NNNet::compute( CTVFlt &inp ) const {
    TVFlt obuf(sizeBuf());
    compute( inp , obuf );
    return obuf.mid( offsetOut() );
}


void NNNet::gradient( CTVFlt &inp , CTVFlt &out , TVFlt &obuf , TVFlt &ibuf , TVFlt &grad  ) const {
//    QTextStream stdo(stdout);

    compute( inp , obuf );

    ibuf.fill( 0 );
    for( nnityp i=0,j=offsetOut() ; i<out.size() ; i++,j++ ) {
        ibuf[j] = obuf[j] - out[i];
//        stdo << "out: " << ibuf[j] << "=" << obuf[j] << "-" << out[i] << "[" << i << "," << j << "]" << endl;
    }

    for( nnityp i=neurons.size()-1 ; i>=0 ; i-- ) {
        CNNNeuron &neuron = neurons[i];
        CTVInt &idx_i = neuron.idx_i;
        CTVInt &idx_w = neuron.idx_w;        
        ibuf[ i + size_i ] *= fderivate( obuf[ i + size_i ] , neuron.actv );
//        stdo << "neu:" << i << " ibuf_idx:" << (i+size_i) << " ibuf:" << ibuf[i+size_i] << " obuf:" << obuf[i+size_i] << endl;
        for( nnityp j=0 ; j<idx_i.size() ; j++ ) {
            ibuf[ idx_i[j] ] += weights[ idx_w[j] ] * ibuf[ i + size_i ];
        }
    }

    for( nnityp i=0 ; i<neurons.size() ; i++ ) {
        CNNNeuron &neuron = neurons[i];
        CTVInt &idx_i = neuron.idx_i;
        CTVInt &idx_w = neuron.idx_w;
        for( nnityp j=0 ; j<idx_i.size() ; j++ ) {
            grad[ idx_w[j] ] += obuf[ idx_i[j] ] * ibuf[ i + size_i ];
        }
    }

}

void NNNet::gradientN( CTVFlt &inp , CTVFlt &out , TVFlt &obuf , TVFlt &grad  ) {
    nncftyp d = 0.0000001;
    nncftyp e = error( inp , out , obuf );
    for( nnityp i=0 ; i<weights.size() ; i++ ) {
        nncftyp copy = weights[i];
        weights[i] += d;
        grad[i] += 0.5 * ( error( inp , out , obuf ) - e ) / d;
        weights[i] = copy;
    }
}

TVFlt NNNet::gradientN( CNNData &data ) {
    TVFlt gr( sizeWeights() );
    nncftyp d = 0.000001;
    nncftyp e = error( data );
    for( nnityp i=0 ; i<weights.size() ; i++ ) {
        nncftyp copy = weights[i];
        weights[i] += d;
        gr[i] = 0.5 * ( error( data ) - e ) / d;
        weights[i] = copy;
    }
    return gr;
}


TVFlt NNNet::gradient( CNNData &data , nncftyp bigPenal ) const {
    nncityp mt = omp_get_max_threads();

    NNVec< TVFlt > grads( mt );
    grads.fill( TVFlt( sizeWeights() , 0 ) );

    TVFlt obufs( sizeBuf() );
    TVFlt ibufs( sizeBuf() );

#pragma omp parallel for firstprivate(obufs,ibufs)
    for( nnityp i=0 ; i<data.size() ; i++ ) {
        nncityp nt = omp_get_thread_num();
        gradient( data[i].getInps() , data[i].getOuts() , obufs , ibufs, grads[nt] );
    }

    TVFlt grad( sizeWeights() , 0 );
    for( nnityp i=0 ; i<mt ; i++ ) {
        for( nnityp j=0 ; j<sizeWeights() ; j++ )
            grad[j] += grads[i][j];
    }

    for( nnityp i=0 ; i<sizeWeights() ; i++ ) {
        grad[i] /= data.size() * size_o;
    }

    // f = w1^2 + w2^2 + ... + wn^2
    // df/dwi = 2wi

    for( nnityp i=0 ; i<sizeWeights() ; i++ ) {
        grad[i] += 2 * weights[i] * bigPenal;
    }

    return grad;
}

// Liczy gradient tylko dla wskazanych numer??w wag
TVFlt NNNet::subGradient( CNNData &data , CTVInt &sub ) const {
    CTVFlt gr1 = gradient( data );
    TVFlt gr2( sub.size() );
    for( nnityp i=0 ; i<sub.size() ; i++ ) {
        gr2[i] = gr1[sub[i]];
    }
    return gr2;
}

static nnftyp fnorm( CTVFlt &v ) {
    nnftyp norm = 0;
    for( nnityp i=0 ; i<v.size() ; i++ ) {
        norm += v[i] * v[i];
    }
    return sqrt(norm);
}

static nnftyp avgNorm( CTVFlt &v ) {
    nnftyp norm = 0;
    for( nnityp i=0 ; i<v.size() ; i++ ) {
        norm += v[i] * v[i];
    }
    return sqrt( norm / v.size() );
}

static void mkNorm( TVFlt &v ) {
    nncftyp n = fnorm(v);
    for( nnityp i=0 ; i<v.size() ; i++ ) {
        v[i] /= n;
    }
}

static TVFlt mkNorm2( CTVFlt &v ) {
    nncftyp n = fnorm(v);
    TVFlt out( v.size() );
    for( nnityp i=0 ; i<v.size() ; i++ ) {
        out[i] = v[i] / n;
    }
    return out;
}


static TVFlt& makeToLearn( TVFlt &gr , CTVFlt &tolearn ) {
    for( nnityp i=0 ; i<std::min(gr.size(),tolearn.size()) ; i++ ) {
        gr[i] *= tolearn[i];
    }
    return gr;
}

void NNNet::rawDescent( CNNData &data , nncityp loops , nncftyp step , CTVFlt &tolearn ) {
    for( nnityp loop = 0 ; loop < loops ; loop ++ ) {
        TVFlt gr = gradient( data );
        makeToLearn( gr , tolearn );
        mkNorm(gr);
        for( nnityp j=0 ; j<sizeWeights() ; j++ ) {
            weights[j] -= step * gr[j];
            if( weights[j] > max_w[j] ) weights[j] = max_w[j];
            if( weights[j] < min_w[j] ) weights[j] = min_w[j];
        }
    }
}

void NNNet::grDescent( CNNData &data , nncityp loops , nncftyp step , CTVFlt &tolearn, nncftyp min_error, nnityp subloops , nnityp show ) {
    QTextStream stdo(stdout);
    stdo.setRealNumberPrecision(8);
    stdo.setRealNumberNotation( QTextStream::FixedNotation );
    nnftyp er1 = error( data );
    nnftyp ng = 0;
    nnityp loop = 0;
    bool work = true;
    stdo << "    loop] " << "     error       step        ||gr||   ||weights||" << endl;
    while( work ) {
        CTVFlt copy = weights;
        rawDescent( data , subloops , step , tolearn );
        nncftyp er2 = error( data );
        if( er2 >= er1 - min_error ) {
            if( subloops > 1 ) {
                subloops = 1;
            } else {
                subloops = 0;
            }
        }
        if( er2 <= er1 ) {
            er1 = er2;
        } else {
            weights = copy;
        }
        ng = avgNorm( gradient(data) );
        if( loop % show == 0 ) {
            stdo << qSetFieldWidth(8) << loop << qSetFieldWidth(0) << "] " << er2 << " " << step << " " << qSetFieldWidth(13) << ng << qSetFieldWidth(0) << " " << qSetFieldWidth(13) << avgNorm(weights) << endl;
        }
        loop ++ ;
        work = subloops==1 || (subloops>0 && loop < loops);
    }
    stdo << qSetFieldWidth(8) << loop << qSetFieldWidth(0) << "] " << er1 << " " << step << " " << qSetFieldWidth(13) << ng << qSetFieldWidth(0) << " " << qSetFieldWidth(13) << avgNorm(weights) << endl;
}

//void NNNet::rawMomentum( CNNData &data , nncityp loops , nncftyp step , TVFlt &p , nncftyp mom,  nncftyp bigpenal, CTVFlt &tolearn) {
//    for( nnityp loop = 0 ; loop < loops ; loop++ ) {
//        TVFlt g = gradient(data,bigpenal);
//        makeToLearn( g , tolearn );
//        mkNorm( g );
//        for( nnityp i=0 ; i<p.size() ; i++ ) {
//            p[i] = p[i] * mom + g[i] * (1.0-mom);
//        }
//        for( nnityp i=0 ; i<p.size() ; i++ ) {
//            weights[i] -= p[i] * step;
//            if( weights[i] > max_w[i] ) weights[i] = max_w[i];
//            if( weights[i] < min_w[i] ) weights[i] = min_w[i];
//        }
//    }
//}

void NNNet::rawMomentum( CNNData &data , nncityp loops , nncftyp step , TVFlt &p , nncftyp mom,  nncftyp bigpenal, CTVFlt &tolearn) {
    TVFlt d(p.size());
    for( nnityp loop = 0 ; loop < loops ; loop++ ) {
        TVFlt g = gradient(data,bigpenal);
        makeToLearn( g , tolearn );
        mkNorm( g );
        for( nnityp i=0 ; i<p.size() ; i++ ) {
            d[i] = p[i] = p[i] * mom + g[i] * (1.0-mom);
        }
        mkNorm( d );
        for( nnityp i=0 ; i<p.size() ; i++ ) {
            weights[i] -= d[i] * step;
            if( weights[i] > max_w[i] ) weights[i] = max_w[i];
            if( weights[i] < min_w[i] ) weights[i] = min_w[i];
        }
    }
}



void NNNet::momentum(CNNData &data , nncityp loops , nncftyp step , nncftyp mom, CTVFlt &tolearn, nncftyp min_error, nncftyp bigpenal, nnityp subloops , nnityp show) {
    QTextStream stdo(stdout);
    stdo.setRealNumberPrecision(8);
    stdo.setRealNumberNotation( QTextStream::FixedNotation );
//    TVFlt p = mkNorm2( gradient( data ) );
    TVFlt p( sizeWeights() , 0 );
    nnftyp er1 = error( data , bigpenal );
    nnftyp ng = 0;
    nnityp fails = 0;
    nnityp loop=0;
    bool work = true;
    stdo << "    loop]      error       step  ||gradient||   ||weights||         ||p||  fails" << endl;
    while( work ) {
        CTVFlt copy = weights;
        rawMomentum( data , subloops , step , p , mom , bigpenal , tolearn );
        nncftyp er2 = error( data , bigpenal );

        if( er2 < er1 - min_error ) {
            fails = 0;
        } else {
            if( fails == 0 )
                p.fill(0);
            fails ++ ;
        }
        if( er2 > er1 ) {
            weights = copy;
        } else {
            er1 = er2;
        }
        ng = avgNorm( gradient(data) );
        if( loop % show == 0 ) {
            stdo << qSetFieldWidth(8) << loop << qSetFieldWidth(0) << "] " << er1 << " " << step << " " << qSetFieldWidth(13) << ng << qSetFieldWidth(0) << " " << qSetFieldWidth(13) << avgNorm(weights) << qSetFieldWidth(0) << " " << qSetFieldWidth(13) << fnorm(p)  << qSetFieldWidth(0) << " " << qSetFieldWidth(6) << fails << endl;
        }
        loop ++ ;
        work = loop < loops && fails < 4;
    }
    stdo << qSetFieldWidth(8) << loop << qSetFieldWidth(0) << "] " << er1 << " " << step << " " << qSetFieldWidth(13) << ng << qSetFieldWidth(0) << " " << qSetFieldWidth(13) << avgNorm(weights) << qSetFieldWidth(0) << " " << qSetFieldWidth(13) << fnorm(p)  << qSetFieldWidth(0) << " " << qSetFieldWidth(6) << fails << endl;
}


void NNNet::momentum2(
    CNNData &learn ,         //MM: Data to learn.
    CNNData &test ,          //MM: Data to test.
    const time_t maxTime ,
    nncityp maxFailsTest,
    nnftyp step,
    nncftyp maxStep,
    nncftyp minStep,
    nncftyp mom,
    CTVFlt &toLearn,
    nncftyp bigPenal,
    nnityp subLoops ,
    NNNet *const best
) {
    QTextStream stdOut(stdout);
    stdOut.setRealNumberPrecision(6);
    stdOut.setRealNumberNotation( QTextStream::FixedNotation );
    TVFlt p( sizeWeights() , 0 );
    nnftyp er1 = error( learn , bigPenal );
    nnftyp testError = 0;
    if( test.size() > 0 ) {
        testError = error( test , bigPenal );
    }
    nnityp fails = 0, success = 0, failsTest = 0;
    TVFlt bestWeights = weights;
    stdOut << "    loop] learn error";
    if( test.size() > 0 ) {
        stdOut << "        test";
        stdOut << "   best_test";
    }
    stdOut<< "           step   ||gradient||   ||weights||         ||p||  fails     time" << endl;
    nnityp loop = 0;
    const time_t start = time(NULL);
    time_t currTime = start;
    time_t lastShow = currTime;
    time_t showTime = 1;
    nnftyp er3 = 0;
    nnftyp stepInc = 1.25;
    while(
        (maxTime==0 || currTime-start <= maxTime)
            &&
        step >= minStep
            &&
        (maxFailsTest==0 || failsTest < maxFailsTest)
    ) {
        loop++;
        currTime = time(NULL);
        CTVFlt copyP = p;
        rawMomentum( learn , subLoops , step , p , mom , bigPenal , toLearn );
        nncftyp er2 = error( learn , bigPenal );

        if( er2 > er1 ) {
            weights = bestWeights;
            step *= 0.42;
            stepInc = 1.0341;
            success = 0;
            fails ++ ;
            for( nnityp i=0 ; i<p.size() ; i++ ) {
                p[i] = copyP[i] * 0.41;
            }
        } else {
            bestWeights = weights;

            if( er2 < er1 ) {
                er1 = er2;
                fails = 0;
                if( ++success >= 1 ) {
                    step *= stepInc;
                }
                if( step > maxStep ) {
                    step = maxStep;
                }
            }

            if( test.size() > 0 ) {
                er3 = error( test , bigPenal );
                if( er3 <= testError ) {
                    failsTest = 0;
                    testError = er3;
                    if( best != nullptr ) {
                        *best = *this;
                    }
                } else {
                    failsTest ++ ;
                }
            }

        }

        if( loop == 1 || currTime - lastShow >= showTime ) {

            stdOut << qSetFieldWidth(8) << loop;
            stdOut << qSetFieldWidth(0) << "] ";
            stdOut << qSetFieldWidth(11) << qSetRealNumberPrecision(8) << er1;
            if( test.size() > 0 ) {
                stdOut << qSetFieldWidth( 0) << " ";
                stdOut << qSetFieldWidth(11) <<  er3;
                stdOut << qSetFieldWidth( 0) << " ";
                stdOut << qSetFieldWidth(11) << testError;
            }
            stdOut << qSetFieldWidth(0) << " ";
            stdOut << qSetFieldWidth(9) << qSetRealNumberPrecision(12) << step;
            stdOut << qSetFieldWidth(0) << " ";
            TVFlt gr = gradient(learn,bigPenal);
            stdOut << qSetFieldWidth(13)<< fnorm( makeToLearn( gr, toLearn ) );
            stdOut << qSetFieldWidth(0) << " ";
            stdOut << qSetFieldWidth(13) << qSetRealNumberPrecision(8) << fnorm(weights);
            stdOut << qSetFieldWidth(0) << " ";
            stdOut << qSetFieldWidth(13) << fnorm(p);
            stdOut << qSetFieldWidth(0) << " ";
            stdOut << qSetFieldWidth(6) << fails;
            stdOut << qSetFieldWidth(0) << " ";
            stdOut << qSetFieldWidth(7) << (currTime-start);
            stdOut << qSetFieldWidth(0) << "s";
            stdOut << endl;

            lastShow = currTime;

            nnityp ds = 0;
            if( currTime - start >= 5000 ) {
                showTime = 500;
                ds       = 100;
            } else if( currTime - start >=3000 ) {
                showTime = 300;
                ds       =  50;
            } else if( currTime - start >=2000 ) {
                showTime = 200;
                ds       =  10;
            } else if( currTime - start >=1000 ) {
                showTime = 100;
                ds       =  10;
            } else if( currTime - start >= 400 ) {
                showTime =  50;
            } else if( currTime - start >= 200 ) {
                showTime =  20;
            } else if( currTime - start >= 100 ) {
                showTime =  10;
            } else if( currTime - start >=  20 ) {
                showTime =   5;
            } else {
                showTime =   2;
            }
//            showTime = 0;
            if( ds ) {
                nncityp d = (currTime - start + showTime) % ds;
                if( d < ds / 2 ) {
                    showTime -= d;
                } else {
                    showTime += ds - d;
                }
            }

        }
    }
    weights = bestWeights;

}


void NNNet::learnRand3(
    CNNData &learn ,         //MM: Data to learn.
    CNNData &test ,          //MM: Data to test.
    const time_t maxTime ,
    nncityp maxFailsTest,
    nncftyp maxStep,
    nncftyp minStep,
    nncftyp bigPenal,
    nncityp rndSeed,
    NNNet *const best,
    const bool fullVerb
) {
    FRnd rnd(rndSeed);
    QTextStream stdOut(stdout);
    stdOut.setRealNumberPrecision(6);
    stdOut.setRealNumberNotation( QTextStream::FixedNotation );
    nnftyp er1 = error( learn , bigPenal );
    nnftyp er2 = 0;
    nnftyp testError = 0;
    if( test.size() > 0 ) {
        testError = error( test , bigPenal );
    }
    nnityp fails = 0, failsTest = 0;
    TVFlt steps( sizeWeights() );
    for( nnityp i=0 ; i<sizeWeights() ; i++ ) {
        steps[i] = ( rnd()&1 ? -1 : +1 ) * rnd.getF( maxStep/10.0 , maxStep);
    }
    stdOut << "    loop] weight learn_error";
    if( test.size() > 0 ) {
        stdOut << "        test";
        stdOut << "  best_error";
    }
    stdOut<< " ||steps||   ||weights||  fails     time" << endl;
    nnityp loop = 0;
    const time_t start = time(NULL);
    time_t currTime = start;
    time_t lastShow = currTime;
    time_t showTime = 1;

    while( true ) {
        loop++;

        for( nnityp i=0 ; i<sizeWeights() ; i++ ) {

            currTime = time(NULL);

            if(maxTime      !=0 && currTime-start > maxTime ) {goto endProc;}
            if(maxFailsTest !=0 && failsTest >= maxFailsTest) {goto endProc;}

            nncftyp copy = weights[i];

            for( nnityp j=0 ; j<10 ; j++ ) {
                if( fabs( steps[i] ) < minStep ) {
                    steps[i] = ( rnd()&1 ? -1 : +1 ) * rnd.getF( minStep , maxStep);
                }
                if( steps[i] < -maxStep ) {steps[i] = -maxStep;}
                if( steps[i] > +maxStep ) {steps[i] = +maxStep;}
                weights[i] += steps[i];
                nncftyp tmp = error(learn,bigPenal);
                if( tmp <= er1 ) {
                    if( tmp <= er1 ) {
                        er1 = tmp;
                        steps[i] *= 2;
                    }
                    if( test.size() > 0 ) {
                        er2 = error( test , bigPenal );
                        if( er2 <= testError ) {
                            failsTest = 0;
                            testError = er2;
                            if( best != nullptr ) {
                                *best = *this;
                            }
                        } else {
                            failsTest ++ ;
                        }
                    }
                    break;
                }
                weights[i] = copy;
                steps[i] *= -0.5;
            }

            if( fullVerb || (loop == 1 && i==0 ) || currTime - lastShow >= showTime ) {

                stdOut << qSetFieldWidth(8) << loop;
                stdOut << qSetFieldWidth(0) << " ";
                stdOut << qSetFieldWidth(6) << i;
                stdOut << qSetFieldWidth(0) << "] ";
                stdOut << qSetFieldWidth(11) << qSetRealNumberPrecision(8) << er1;
                if( test.size() > 0 ) {
                    stdOut << qSetFieldWidth( 0) << " ";
                    stdOut << qSetFieldWidth(11) <<  er2;
                    stdOut << qSetFieldWidth( 0) << " ";
                    stdOut << qSetFieldWidth(11) << testError;
                }
                stdOut << qSetFieldWidth(0) << " ";
                stdOut << qSetFieldWidth(9) << qSetRealNumberPrecision(6) << avgNorm(steps);
                stdOut << qSetFieldWidth(0) << " ";
                stdOut << qSetFieldWidth(13) << avgNorm(weights);
                stdOut << qSetFieldWidth(0) << " ";
                stdOut << qSetFieldWidth(6) << fails;
                stdOut << qSetFieldWidth(0) << " ";
                stdOut << qSetFieldWidth(7) << (currTime-start);
                stdOut << qSetFieldWidth(0) << "s";
                stdOut << endl;

                lastShow = currTime;
                if( currTime - start >= 2000 ) {
                    showTime = 200;
                } else if( currTime - start >= 1000 ) {
                    showTime = 100;
                } else if( currTime - start >= 500 ) {
                    showTime =  50;
                } else if( currTime - start >= 300 ) {
                    showTime =  25;
                } else if( currTime - start >= 100 ) {
                    showTime =  20;
                } else {
                    showTime =  10;
                }
            }
        }
    }
    endProc:
    ;
}

void NNNet::multiInit(
    CNNData &data,
    const time_t maxTime,
    nncityp maxLoops,
    nncityp minWeight,
    nncityp maxWeight,
    const bool rndWeight,
    const bool rndIdxWeight,
    const bool rndInput,
    nncftyp bigPenal,
    nncityp rndSeed,
    nncityp bestSize,
    const bool show
) {
    struct RngValue {
        nnftyp range;
        nnftyp value;
        nnityp use = 1;
        RngValue() {}
        RngValue( nncftyp range, nncftyp value) : range(range), value(value) {
        }
        bool operator < (const RngValue &other) const {
            return value < other.value;
        }
        bool operator == (const RngValue &other) const {
            return fabs( this->range - other.range ) <= std::numeric_limits<nnftyp>::epsilon() * std::max( fabs(this->range) , fabs(other.range) );
        }
    };
    QTextStream stdOut(stdout);
    stdOut.setRealNumberPrecision(6);
    stdOut.setRealNumberNotation( QTextStream::FixedNotation );

    FRnd rnd(rndSeed);
    QVector<RngValue> bestRanges;
    const time_t start =  time(NULL);
    time_t lastShow = start;
    time_t showTime = 5;
    nnftyp value = error(data);
    NNNet bestNN = *this;
    for( int loop=1 ; ( maxLoops <= 0 || loop <= maxLoops ) && ( maxTime <= 0 || (time(nullptr) - start) < maxTime ) ; loop++ ) {
        nnftyp range;
        if( rndWeight ) {
            if( bestRanges.size() > 0 && rnd() % 2 ) {
                range = bestRanges[ (rnd() % bestRanges.size()) /  8 ].range;
            } else if( bestRanges.size() > 0 && rnd() % 2 ) {
                range = bestRanges[ (rnd() % bestRanges.size()) /  4 ].range;
            } else if( bestRanges.size() > 0 && rnd() % 2 ) {
                range = bestRanges[ (rnd() % bestRanges.size()) /  2 ].range;
            } else if( bestRanges.size() > 0 && rnd() % 2 ) {
                range = bestRanges[ (rnd() % bestRanges.size()) /  1 ].range;
            } else {
//                range = rnd.getF( minWeight , maxWeight );
                nncityp diffRnd = 10000;
                range = ( (nnftyp)( rnd() % (diffRnd+1) ) / diffRnd ) * ( maxWeight - minWeight ) + minWeight;
            }
            randWeights( rnd , -range , +range );
        }
        if( rndInput ) {
            randInputs(rnd);
        }
        if( rndIdxWeight ) {
            randIdxW( rnd );
        }
        nncftyp newValue = error( data, bigPenal );
        if( rndWeight ) {
            nncityp idx = bestRanges.indexOf( RngValue(range,newValue) );
            if( idx < 0 ) {
                bestRanges += RngValue(range,newValue);
            } else {
                bestRanges[idx].value *= bestRanges[idx].use;
                bestRanges[idx].value += newValue;
                bestRanges[idx].value /= ++bestRanges[idx].use;
            }
            std::sort( bestRanges.begin() , bestRanges.end() );
            if( bestRanges.size() > bestSize ) {
                bestRanges.pop_back();
            }
        }
        if( newValue < value ) {
            value = newValue;
            bestNN = *this;
        }

        if( show ) {
            const time_t currTime = time(NULL);
            if( currTime - lastShow >= showTime ) {
                stdOut << "loop=" << loop << " time=" << (time(nullptr) - start) << "s value=" << value << endl;
                const bool fullShow = false;
                if( fullShow ) {
                    for( int i=0 ; i<bestRanges.size() ; i++ ) {
                        stdOut << i << "] " << bestRanges[i].value << " " << bestRanges[i].range << " " << bestRanges[i].use << endl;
                    }
                    stdOut << "---------------------------------------" << endl;
                }
                if( currTime - start > 7200 ) {
                    showTime = 120;
                } else if( currTime - start > 3600 ) {
                    showTime =  60;
                } else if( currTime - start > 1800 ) {
                    showTime =  30;
                } else if( currTime - start > 600 ) {
                    showTime =  20;
                } else if( currTime - start > 120 ) {
                    showTime =  15;
                } else if( currTime - start >  60 ) {
                    showTime =  10;
                } else {
                    showTime =   5;
                }
                lastShow = currTime;
            }
        }

        if( loop == bestSize*3 || loop % 10000 == 0 ) {
        }
    }
    *this = bestNN;
}




// Reset hesjanu
static void bfgsResetHes( TVFlt &hes ) {
    hes.fill(0);
    nncityp size = sqrt( hes.size() );
    EXCP( size  * size  == hes.size() );
    for( nnityp i=0 ; i<size  ; i++ ) {
        hes[ i + i * size ] = 1;
    }
}

// Buduje znormalizowany kierunek z hesjanu i gradientu.
static TVFlt bfgsDir( TVFlt &hes, TVFlt &g ) {
    nncityp size = g.size();
    TVFlt p( size );
    for( nnityp i=0 ; i<size ; i++ ) {
        p[ i ] = 0;
        for( nnityp j=0 ; j<size ; j++ ) {
            p[ i ] += hes[ i * size + j ] * g[ j ];
        }
    }
    mkNorm( p );
    return p;
}

static void minDir( TVFlt &weights , CTVFlt &p , CTVInt &sub , nncftyp step ) {
//    QTextStream stdo(stdout);
    for( nnityp i=0 ; i<sub.size() ; i++ ) {
//        stdo << p[i] << " " << weights[i] << " " << sub[i] << " " << step << endl;
        weights[ sub[i] ] -= p[i] * step;
    }
}

void NNNet::bfgs( CNNData &data , CTVInt &sub , nncityp loops, nncftyp step ) {
    QTextStream stdo(stdout);
    nncityp size = sub.size();                   // Skr??t do rozmiaru pod zbioru optymalizowanych wag.
    TVFlt g = subGradient( data , sub );         // Gradient
    nnftyp ng = fnorm(g);                        // Norma gradientu.
    TVFlt pg(size);                              // Poprzedni gradient
    TVFlt s(size), r(size), rv(size), vr(size);  // Wektory pomocnicze.
    TVFlt hes( size * size );                    // Hesjan.
    nnftyp er1 = error( data );                  // B????d.
    TVFlt best = weights;                        // Kopia najlepszych wag.
    nnityp fail  =  0;                           // Ilo???? nieudanych krok??w.
    nnityp reset = 10;                           // Pewna odleg??o???? od resetu
    bfgsResetHes( hes );                         // Reset hesjanu.
    for( nnityp loop=0 ; loop<loops && ng > 1E-6 ; loop++ ) {
        CTVFlt prev = weights;                   // kopia wag
        CTVFlt p = bfgsDir( hes , g );           // znormalizowany kierunek zgodny z bfgs
        minDir( weights , p , sub , step / reset ); // Zanim hesjan dobrze si?? nie uaktualni, zmniejszamy krok.
        nncftyp er2 = error( data );

        if( er2 < er1 - 1E-6 ) {
            er1 = er2;
            best = weights;
            fail = 0;
        } else {
            fail ++ ;
        }

        pg = g;
        g = subGradient( data, sub );
        ng = fnorm( g );
        stdo << loop << " " << er1 << " " << ng << " " << reset << " " << fail << endl;

        if( fail >= 6 )
            break;

        if( fail >= 5 ) {
            bfgsResetHes( hes );
            weights = best;
            reset = 10;
            continue;
        }

        for( nnityp i=0 ; i<size ; i++ ) {
            s[i] = weights[sub[i]] - prev[sub[i]];
            r[i] = g[i] - pg[i];
        }

        nnftyp rvr = 0;
        nnftyp sr = 0;
        for(nnityp i=0 ; i<size ; i++ ) {
            rv[i] = vr[i] = 0;
            for(nnityp j=0 ; j<size ; j++) {
                rv[i] += hes[ j * size + i ] * r[ j ] ;
                vr[i] += hes[ i * size + j ] * r[ j ];
            }
            rvr += rv[i] * r[i];
            sr  += s[i] * r[i];
        }

        if( fabs(sr) > 1E-8 ) {
            nncftyp rvr_sr = ( 1 + rvr / sr ) / sr;
            for(nnftyp i=0 ; i<size ; i++ ) {
                for(nnftyp j=0 ; j<size ; j++ ) {
                    hes[i*size+j] += rvr_sr * s[i]*s[j] - (s[i] * rv[j] + vr[i] * s[j]) / sr;
                }
            }
        } else {
            bfgsResetHes( hes );
            reset = 10;
            continue;
        }

        if( reset > 1 )
            reset -- ;
    }
    weights = best;
}

void NNNet::randInputs( FRnd &rnd ) {
    for( nnityp i=0 ; i<neurons.size() ; i++ ) {
        NNNeuron &n = neurons[i];
        TVInt &idx_i = n.idx_i;
        TVInt unique;
        for( nnityp j=0 ; j<idx_i.size() ; j++ ) {
            nnityp r;
            for( nnityp k=0 ; k<1000 ; k++ ) {
                r = rnd.getI( n.min_i , n.max_i );
                if( ! unique.contains(r) ) {
                    break;
                }
            }
            unique.append( r );
            idx_i[j] = r;
        }
        std::sort( idx_i.begin(), idx_i.end(), [](nncityp &a,nncityp &b){ return a < b;} );
    }
}

//void NNNet::randIdxW(FRnd &rnd, nncftyp min, nncftyp max, nncityp maxTry) {
//    for( nnityp i=0 ; i<neurons.size() ; i++ ) {
//        NNNeuron &n = neurons[i];
//        TVInt &idx_w = n.idx_w;
//        for( nnityp j=0 ; j<idx_w.size() ; j++ ) {
//            for( nnityp k=0 ; k<maxTry ; k++ ) {
//                idx_w[j] = rnd.getI( n.min_w , n.max_w );
//                nncftyp w = weights[idx_w[j]];
//                if( max - min < 1E-6 || ( w >= min && w <= max  ) ) {
//                    break;
//                }
//            }
//        }
//    }
//}

void NNNet::randIdxW(FRnd &rnd, nncftyp min, nncftyp max, nncityp maxTry) {
    for( nnityp i=0 ; i<neurons.size() ; i++ ) {
        NNNeuron &n = neurons[i];
        TVInt &idx_w = n.idx_w;
        TVInt unique;
        for( nnityp j=0 ; j<idx_w.size() ; j++ ) {
            nnityp r;
            for( nnityp k=0 ; k<maxTry ; k++ ) {
                r = rnd.getI( n.min_w , n.max_w );
                nncftyp w = weights[r];
                if( ! unique.contains(r) && ( max - min < 1E-6 || ( w >= min && w <= max  ) ) ) {
                    break;
                }
            }
            unique.append( r );
            idx_w[j] = r;
        }
    }
}

void NNNet::randWeights( FRnd &rnd , nncftyp min , nncftyp max ) {
    for( nnityp i=0 ; i<sizeWeights() ; i++ ) {
        if( max-min > 0.000001 ) {
            weights[i] = rnd.getF( min , max );
        } else if( max_w[i] - min_w[i] > 0.000001 ){
            weights[i] = rnd.getF( min_w[i] , max_w[i] );
        }
        if( weights[i] < min_w[i] ) {
            weights[i] = min_w[i];
        }
        if( weights[i] > max_w[i] ) {
            weights[i] = max_w[i];
        }
    }
}

int NNNet::doubleRndIdxWeightForce(
    CNNData      &data ,
    const time_t maxTime,
    nncityp      maxLoops,
    nncityp      maxNotIncrese,
    nncityp      rndSeed,
    const bool   show
) {
    const time_t start = time(NULL);
    QTextStream stdOut(stdout);
    stdOut.setRealNumberPrecision(13);
    stdOut.setRealNumberNotation( QTextStream::FixedNotation );
    time_t lastShow = start;
    FRnd rnd(rndSeed);
    nnityp sumIncrease = 0;
    nnityp notIncrese = 0;
    nnityp callEval = 1;
    time_t showTime = 0;
    nnftyp er = error(data);

    stdOut << "    loop ]           error   callEval   sumIncrease    time " << endl;
    for( nnityp loop=1 ; (maxNotIncrese <= 0 ||  notIncrese < maxNotIncrese ) && (maxLoops <= 0 || loop <= maxLoops) && (maxTime <= 0 || time(NULL)-start < maxTime); loop++ ) {
        nnityp neuronI,neuronJ,idxWeightI,idxWeightJ;
        do {
            neuronI = rnd() % neurons.size();
            neuronJ = rnd() % neurons.size();
            idxWeightI = rnd() % neurons[neuronI].idx_w.size();
            idxWeightJ = rnd() % neurons[neuronJ].idx_w.size();
        } while( neuronI == neuronJ && idxWeightI == idxWeightJ );
        nnityp bestI = neurons[neuronI].idx_w[ idxWeightI ];
        nnityp bestJ = neurons[neuronJ].idx_w[ idxWeightJ ];
        bool increse = false;
        for( nnityp i = neurons[neuronI].min_w ; i <= neurons[neuronI].max_w ; i++ ) {
            neurons[neuronI].idx_w[ idxWeightI ] = i;
            for( nnityp j = neurons[neuronJ].min_w ; j <= neurons[neuronJ].max_w ; j++ ) {
                neurons[neuronJ].idx_w[ idxWeightJ ] = j;
                nncftyp tmp = error(data); callEval ++ ;
                if( tmp <= er ) {
                    if( tmp < er ) {
                        increse = true;
                        sumIncrease ++ ;
                        er = tmp;
                    }
                    bestI = i;
                    bestJ = j;
                }
                if( show ) {
                    const time_t currTime = time(NULL);
                    if( currTime - lastShow >= showTime ) {
                        stdOut << qSetFieldWidth(8)  << loop;
                        stdOut << qSetFieldWidth(0)  << " ] ";
                        stdOut << qSetFieldWidth(15) << er;
                        stdOut << qSetFieldWidth(0)  << " ";
                        stdOut << qSetFieldWidth(10)  << callEval;
                        stdOut << qSetFieldWidth(0)  << " ";
                        stdOut << qSetFieldWidth(13) << sumIncrease;
                        stdOut << qSetFieldWidth(0)  << " ";
                        stdOut << qSetFieldWidth(7)  << (currTime-start);
                        stdOut << qSetFieldWidth(0)  << "s" << endl;
                        if( currTime - start > 7200 ) {
                            showTime = 120;
                        } else if( currTime - start > 3600 ) {
                            showTime =  60;
                        } else if( currTime - start > 1800 ) {
                            showTime =  30;
                        } else if( currTime - start > 600 ) {
                            showTime =  20;
                        } else if( currTime - start > 120 ) {
                            showTime =  15;
                        } else if( currTime - start >  60 ) {
                            showTime =  10;
                        } else {
                            showTime =   5;
                        }
                        lastShow = currTime;
                    }
                }
            }
        }
        neurons[neuronI].idx_w[ idxWeightI ] = bestI;
        neurons[neuronJ].idx_w[ idxWeightJ ] = bestJ;
        if( increse ) {
            notIncrese = 0;
        } else {
            notIncrese ++ ;
        }
    }
    return sumIncrease;
}


int NNNet::forceIdx( CNNData &data , nncftyp bigPenal, const time_t maxTime, const int maxLoops, const bool show ) {
    int increse = 0;
    QTextStream stdo(stdout);
    stdo.setRealNumberPrecision(8);
    stdo.setRealNumberNotation( QTextStream::FixedNotation );
    nnftyp er = error(data);
    const time_t start = time(nullptr);
    for( nnityp loop=0 ; ( maxTime <= 0 || (time(nullptr)-start) <= maxTime ) && (maxLoops <= 0 || loop < maxLoops) ; loop ++ ) {
        bool work = false;
        for( nnityp i=0 ; i<neurons.size() ; i++ ) {
            NNNeuron &n = neurons[ i ];
            TVInt idx_w = n.idx_w;
            for( nnityp j=0 ; j<idx_w.size() ; j++ ) {
                nnityp best = n.idx_w[j];
                for( nnityp k=0 ; k<sizeWeights() ; k++ ) {
                    n.idx_w[j] = k;
                    nncftyp tmp = error(data,bigPenal);
                    if( tmp < er ) {
                        increse ++ ;
                        work = true;
                        er = tmp;
                        best = k;
                        if( show ) {
                            stdo << loop << "] " << i << " " << j << " " << k << " " << er << endl;
                        }
                    }
                }
                n.idx_w[j] = best;
            }
        }
        if( ! work )
            break;
    }
    return increse;
}


int NNNet::forceIdx2(
    CNNData      &data ,
    nncftyp      bigPenal,
    const time_t maxTime,
    nncityp      maxLoops,
    const bool   show
) {
    const time_t start = time(NULL);
    int increse = 0;

    QTextStream stdo(stdout);

    stdo.setRealNumberPrecision(10);
    stdo.setRealNumberNotation( QTextStream::FixedNotation );

    nnftyp er = error(data);
    bool isIncrese = true;

    for( nnityp loop=1 ; isIncrese && ( maxTime <= 0 || time(NULL)-start < maxTime) && (maxLoops <= 0 || loop <= maxLoops); loop++ ) {

        isIncrese = false;

        for( nnityp i=0 ; i<neurons.size()  && ( maxTime <= 0 || time(NULL)-start < maxTime) ; i++ ) {
            NNNeuron &n = neurons[ i ];
            TVInt &idx_i = n.idx_i;
            TVInt &idx_w = n.idx_w;
            assert( idx_i.size() == idx_w.size() );
            for( int j=0 ; j < idx_i.size() ; j++ ) {
                nnityp best_i = idx_i[j];
                nnityp best_w = idx_w[j];
                for( nnityp l=n.min_i ; l<=n.max_i ; l++ ) {
                    idx_i[ j ] = l;
                    for( nnityp k=n.min_w ; k<=n.max_w ; k++ ) {
                        idx_w[ j ] = k;
                        nncftyp tmp = error(data,bigPenal);
                        if( tmp < er ) {
                            isIncrese = true ;
                            increse ++ ;
                            er = tmp;
                            best_i = l;
                            best_w = k;
                            if( show ) {
                                stdo << loop << "] " << i << " " << j << " " << k << " " << er << endl;
                            }
                        }
                    }
                }
                idx_i[j] = best_i;
                idx_w[j] = best_w;
            }
        }
    }
    return increse;
}


int NNNet::forcePairIdx( CNNData &data , nncftyp bigPenal, nncityp loops, const bool show , nncityp rndSeed ) {
    FRnd rnd(rndSeed);
    int increse = 0;
    QTextStream stdo(stdout);
    stdo.setRealNumberPrecision(8);
    stdo.setRealNumberNotation( QTextStream::FixedNotation );
    nnftyp er = error(data);

    bool work = true;
    for( nnityp loop=0 ; loop<loops && work ; loop ++ ) {
        work = false;

        NNVec<nnityp*> idxp;
        for( nnityp i=0 ; i<neurons.size() ; i++ ) {
            NNNeuron &n = neurons[ i ];
            for( nnityp j=0 ; j<n.idx_w.size() ; j++ ) {
                idxp.append( &n.idx_w[j] );
            }
        }

        while( idxp.size() >= 2 ) {
            if( show ) {
                stdo << "loop=" << loop << " " << idxp.size() << endl;
            }
            nnityp *p1 = idxp.takeAt( rnd() % idxp.size() );
            nnityp *p2 = idxp.takeAt( rnd() % idxp.size() );
            nnityp b1 = *p1;
            nnityp b2 = *p2;
            for( nnityp i=0 ; i<sizeWeights() ; i++ ) {
                *p1 = i;
                for( nnityp j=0 ; j<sizeWeights() ; j++ ) {
                    *p2 = j;
                    nncftyp tmp = error(data,bigPenal);
                    if( tmp < er ) {
                        increse ++ ;
                        work = true;
                        er = tmp;
                        b1 = *p1;
                        b2 = *p2;
                        if( show ) {
                            stdo << loop << "] " << i << " " << j << " " << er << endl;
                        }
                    }
                }
                *p1 = b1;
                *p2 = b2;
            }
        }

    }
    return increse;
}


int NNNet::subForceIdx(
    NNData &data,
    nncftyp bigPenal,
    CTVInt &parts,
    nncityp maxTime,
    nncityp maxFails,
    nncityp rndSeed,
    const bool show
) {
    FRnd rnd(rndSeed);

    nnityp sumParts = 0;
    for( nnityp i=0 ; i<parts.size() ; i++ ) {
        sumParts += parts[i];
    }

    if( sumParts > data.size() ) {
        abort();
    }

    TVFlt ers( parts.size() );
    TVFlt tmpErs( parts.size() );

    int increse = 0;
    QTextStream stdOut(stdout);
    stdOut.setRealNumberPrecision(6);
    stdOut.setRealNumberNotation( QTextStream::FixedNotation );

    const time_t start = time(NULL);
    time_t lastError = start;
    time_t currTime  = start;
    nnftyp buffError = error( data , bigPenal );
    nnityp fails = 0;

    for( nnityp loop=0 ; ( maxFails == 0 || fails < maxFails ) && ( maxTime == 0 || (currTime - start) <= maxTime ) ; loop ++ ) {
        data.shuffle(rnd());
        nnityp sum = 0;
        for( nnityp i=0 ; i<parts.size() ; i++ ) {
            ers[i] = error( data , bigPenal , sum , sum + parts[i] );
            sum += parts[i];
        }
        if( sum != sumParts ) {
            abort();
        }

        bool work = false;

        for( nnityp i=0 ; i<neurons.size() && ( maxTime == 0 || (currTime - start) <= maxTime ) ; i++ ) {
            NNNeuron &n = neurons[i];
            TVInt idx_w = n.idx_w;
            for( nnityp j=0 ; j<idx_w.size() ; j++ ) {                
                nnityp best = n.idx_w[j];
                nncityp oldBest = best;
                for( nnityp k=n.min_w ; k<=n.max_w ; k++ ) {
                    n.idx_w[j] = k;
                    sum = 0;
                    for( nnityp i=0 ; i<parts.size() ; i++ ) {
                        tmpErs[i] = error( data, bigPenal , sum , sum + parts[i]);
                        if( tmpErs[i] >= ers[i] ) {
                            break;
                        }
                        sum += parts[i];
                    }
                    if( sumParts == sum ) {
                        ers = tmpErs;
                        increse ++ ;
                        work = true;
                        best = k;
                    }
                }
                n.idx_w[j] = best;
                currTime = time(NULL);
                if( oldBest != best && show ) {
                    if( currTime - lastError >= 3 ) {
                        lastError = currTime;
                        buffError = error( data, bigPenal );
                    }
                    stdOut << qSetFieldWidth(4) << loop;
                    stdOut << qSetFieldWidth(0) << ") ";
                    stdOut << qSetFieldWidth(4) << i;
                    stdOut << qSetFieldWidth(0) << " ";
                    stdOut << qSetFieldWidth(4) << j;
                    stdOut << qSetFieldWidth(0) << " ";
                    stdOut << qSetFieldWidth(4) << best;
                    stdOut << qSetFieldWidth(0) << " ";
                    stdOut << qSetFieldWidth(0) << "[";
                    stdOut << qSetFieldWidth(9) << buffError;
                    stdOut << qSetFieldWidth(0) << "] ";
                    for( nnityp i=0 ; i<ers.size() ; i++ ) {
                        stdOut << qSetFieldWidth(9) << ers[i];
                        stdOut << qSetFieldWidth(0) << " ";
                    }
                    stdOut << qSetFieldWidth(0) << " ";
                    stdOut << qSetFieldWidth(0) << (currTime-start);
                    stdOut << qSetFieldWidth(0) << "s";
                    stdOut << endl;
                }
            }
        }
        if( ! work ) {
            fails ++ ;
        }
    }
    return increse;
}

void NNNet::toUniqueWeights(nncityp skip) {
    PEXCP( skip <= sizeWeights() );
    TVFlt newWeights;    
    TVFlt newMin;
    TVFlt newMax;
    for( nnityp i=0 ; i<skip ; i++ ) {
        newWeights.append( weights[i] );
        newMin.append( min_w[ i ] );
        newMax.append( max_w[ i ] );
    }
    for( auto n=neurons.begin() ; n!=neurons.end() ; ++n ) {
        TVInt &idx_w = n->idx_w;
        for( auto w=idx_w.begin() ; w!=idx_w.end() ; ++w ) {
            newWeights.append( weights[ *w ] );
            newMin.append( min_w[ *w ] );
            newMax.append( max_w[ *w ] );
            *w = newWeights.size()-1;
        }
    }
    for( auto n=neurons.begin() ; n!=neurons.end() ; ++n ) {
        n->min_w = 0;
        n->max_w = newWeights.size()-1;
    }

    weights = newWeights;
    min_w   = newMin;
    max_w   = newMax;
}

void NNNet::setMinWeights(nncftyp min) {
    PEXCP( min_w.size() == sizeWeights() );
    for( nnityp i=0 ; i<sizeWeights() ; i++ ) {
        min_w[i] = min;
    }
}

void NNNet::setMaxWeights(nncftyp max) {
    PEXCP( max_w.size() == sizeWeights() );
    for( nnityp i=0 ; i<sizeWeights() ; i++ ) {
        max_w[i] = max;
    }
}


//MM: Uczenie poprzez losow?? zmian?? idnex??w wag (a nie samych wag).
void NNNet::learnRndIdx(
    CNNData &data ,            //MM: dane ucz??ce
    const time_t maxTime ,     //MM: maksymalna ilo???? czasu
    nncityp maxNotLearn ,      //MM: maksymalna ilo???? iteracji bez minimalnego spadku b????du
    nncftyp minError ,         //MM: warto???? minimalnego spadku b????du
    nncftyp bigpenal ,         //MM: kara za du??e wagi
    nncityp rndSeed ,          //MM: zarodek liczb losowych
    nncftyp pBack              //MM: prawdopodobie??stwo powrotu do lepszego rozwi??zania
) {
    FRnd rnd(rndSeed);
    QTextStream stdOut(stdout);
    stdOut.setRealNumberPrecision(10);
    stdOut.setRealNumberNotation( QTextStream::FixedNotation );
    nnftyp er = error(data, bigpenal );
    NNNet best = *this;
    const time_t start = time(NULL);
    time_t currTime = time(NULL);
    time_t lastShow = currTime;
    time_t lastLearn = currTime;
    time_t showTime = 1;
    for( nnityp loop=0 ; currTime <= maxTime && (maxNotLearn==0 || (currTime-lastLearn) <= maxNotLearn) ; loop ++ ) {
        if( (loop&0xF) == 0 ) {
            currTime = time(NULL);
        }
        NNNeuron &n = neurons[ rnd.getI( neurons.size() ) ];
        if( n.min_w == n.max_w ) {
            continue;
        }
        nncityp rndW = rnd.getI() % n.idx_w.size();
        nncityp copy = n.idx_w[ rndW ];
        do {
            n.idx_w[ rndW ] = rnd.getI( n.min_w , n.max_w );
        } while( n.idx_w[ rndW ] == copy );
        nncftyp tmp = error( data , bigpenal );
        if( tmp > er ) {
            if( rnd.getF() < pBack ) {
                *this = best;
            }
        } else {
            best = *this;
            if( tmp < er - minError ) {
                lastLearn = currTime;
            }
            if( tmp < er ) {
                er = tmp;
            }
        }
        {
            if( loop == 0 || currTime - lastShow >= showTime ) {
                stdOut << qSetFieldWidth(8)  << loop;
                stdOut << qSetFieldWidth(0)  << " ] ";
                stdOut << qSetFieldWidth(13) << er;
                stdOut << qSetFieldWidth(0)  << " ";
                stdOut << qSetFieldWidth(6)  << (currTime-lastLearn);
                stdOut << qSetFieldWidth(0)  << "s ";
                stdOut << qSetFieldWidth(6)  << (currTime-start);
                stdOut << qSetFieldWidth(0)  << "s" << endl;

                lastShow = currTime;
                if( currTime - start > 7200 ) {
                    showTime =  60;
                } else if( currTime - start > 3600 ) {
                    showTime =  30;
                } else if( currTime - start > 1800 ) {
                    showTime =  20;
                } else if( currTime - start > 600 ) {
                    showTime =  10;
                } else if( currTime - start > 120 ) {
                    showTime =   5;
                } else if( currTime - start >  60 ) {
                    showTime =   2;
                } else {
                    showTime =   1;
                }
            }
        }
    }
    *this = best;
}

void NNNet::appendWeight(nncftyp weight, nncftyp min_w, nncftyp max_w) {
    this->weights.append( weight );
    this->min_w.append( min_w );
    this->max_w.append( max_w );
}

void NNNet::extend(nncftyp weight, nncftyp min_w, nncftyp max_w) {
    appendWeight(weight,min_w,max_w);
    for( nnityp i=0 ; i<neurons.size() ; i++ ) {
        neurons[i].min_w = 0;
        neurons[i].max_w = weights.size()-1;
    }
}

void NNNet::learnRand1(
    CNNData &data ,
    nncftyp bigPenal,
    nncityp maxTime ,
    nncityp maxnoti,
    nncftyp mindesc,
    nncftyp min_str,
    nncftyp max_str,
    nncityp seed,
    const bool weights,
    const bool idx_weights,
    const bool idx_input
) {
    QTextStream stdo(stdout);
    stdo.setRealNumberPrecision(8);
    stdo.setRealNumberNotation(QTextStream::FixedNotation);
    FRnd rnd(seed);
    NNNet best = *this;
    nnftyp er = error( data , bigPenal );
    nnityp noti = 0;
    nnftyp strenght = max_str;
    nnftyp decay = pow( min_str / max_str , 1.0 / maxTime );
    const time_t start = time(NULL);
    time_t currTime = start;
    time_t lastShow = currTime;
    time_t showTime = 1;
    for( nnityp loop=1 ; currTime - start <= maxTime && (maxnoti==0 || noti < maxnoti) ; loop++ ) {
        chaos( rnd , strenght , weights, idx_weights, idx_input );
        nncftyp tmp = error( data , bigPenal  );
        if( tmp <= er ) {
            if( tmp < er - mindesc ) {
                noti = 0;
            }
            er = tmp;
            best = *this;
        } else {
            if( rnd.getF() < 0.40 ) {
                *this = best;
            }
            noti ++ ;
        }
        if( (loop & 0x0) == 0 ) {
            const time_t tmp = time(NULL);
            while( currTime < tmp) {
                strenght *= decay;
                currTime++;
            }
        }
        if( loop == 1 || currTime - lastShow >= showTime ) {

            stdo << qSetFieldWidth(8)  << loop;
            stdo << qSetFieldWidth(0)  << "] ";
            stdo << qSetFieldWidth(10) << er;
            stdo << qSetFieldWidth(0)  << " " << qSetFieldWidth(7) << noti;
            stdo << qSetFieldWidth(0)  << " " << qSetFieldWidth(10) << strenght;
            stdo << qSetFieldWidth(0)  << " " << qSetFieldWidth(0) << (currTime-start);
            stdo << qSetFieldWidth(0)  << "s";
            stdo << endl;

            lastShow = currTime;
            nnityp ds = 0;
            if( currTime - start >= 5000 ) {
                showTime = 500;
                ds       = 100;
            } else if( currTime - start >=3000 ) {
                showTime = 300;
                ds       =  50;
            } else if( currTime - start >=2000 ) {
                showTime = 200;
                ds       =  10;
            } else if( currTime - start >=1000 ) {
                showTime = 100;
                ds       =  10;
            } else if( currTime - start >= 400 ) {
                showTime =  50;
            } else if( currTime - start >= 200 ) {
                showTime =  20;
            } else if( currTime - start >= 100 ) {
                showTime =  10;
            } else if( currTime - start >=  20 ) {
                showTime =   5;
            } else {
                showTime =   2;
            }
            if( ds ) {
                nncityp d = (currTime - start + showTime) % ds;
                if( d < ds / 2 ) {
                    showTime -= d;
                } else {
                    showTime += ds - d;
                }
            }

        }

    }
    *this = best;
}

void NNNet::annealing(
    CNNData &data ,
    nncftyp bigPenal,
    nncityp maxTime,
    nncftyp minStrength,
    nncftyp maxStrength,
    nncftyp pBack,
    nnityp  rndSeed
) {
    QTextStream stdOut(stdout);
    stdOut.setRealNumberPrecision(8);
    stdOut.setRealNumberNotation(QTextStream::FixedNotation);
    FRnd rnd(rndSeed);
    TVFlt bestWeights = weights;
    nnftyp er = error(data,bigPenal);
    const time_t start = time(NULL);
    time_t currTime = start;
    time_t lastShow = start;
    time_t showTime = 0;
    nnftyp strenght = maxStrength;
    nnftyp decay = pow( minStrength / maxStrength , 1.0 / maxTime );

    stdOut << "      loop   strength    best_error    curr_error    time" << endl;

    for( nnityp loop=1 ; (currTime-start) <= maxTime ; loop++ ) {

        if( (loop & 0xF) == 0 ) {
            const time_t tmp = time(NULL);
            while( currTime < tmp) {
                strenght *= decay;
                currTime++;
            }
        }

        for( nnityp i=0 ; i<weights.size() ; i++ ) {
            rnd.chaos( weights[i] , min_w[i] , max_w[i] , strenght );
        }

        nncftyp tmp = error(data,bigPenal);
        if( tmp <= er ) {
            er = tmp;
            bestWeights = weights;
        } else if( rnd.getF() < pBack ) {
            weights = bestWeights;
        }

        if( currTime - lastShow >= showTime ) {
            stdOut << qSetFieldWidth(9)  << loop;
            stdOut << qSetFieldWidth(0)  << "] ";
            stdOut << qSetFieldWidth(10) << strenght;
            stdOut << qSetFieldWidth(0)  << " ";
            stdOut << qSetFieldWidth(13) << er;
            stdOut << qSetFieldWidth(0)  << " ";
            stdOut << qSetFieldWidth(13) << tmp;
            stdOut << qSetFieldWidth(0)  << " ";
            stdOut << qSetFieldWidth(6)  << (currTime-start);
            stdOut << qSetFieldWidth(0)  << "s";
            stdOut << endl;

            lastShow = currTime;
            if( currTime - start > 7200 ) {
                showTime = 120;
            } else if( currTime - start > 3600 ) {
                showTime =  60;
            } else if( currTime - start > 1800 ) {
                showTime =  30;
            } else if( currTime - start > 600 ) {
                showTime =  20;
            } else if( currTime - start > 120 ) {
                showTime =  10;
            } else if( currTime - start >  60 ) {
                showTime =   3;
            } else {
                showTime =   1;
            }
        }

    }
    weights = bestWeights;
}


void NNNet::learnRand2(
    NNData  &data,         //MM: Learn and test data.
    CTVInt  &parts,        //MM: Sizes to divide data into learning sets and test set. The remaining data is a test set. If the test set is empty ( sum of parts = data.size ), then testing is not included in the training process.
    nncityp multi,
    nncityp loops,
    nncityp subLoops,
    nncityp notInc,
    nncftyp bigPenal,
    nncftyp minDesc,
    nncftyp minStrength,
    nncftyp maxStrength,
    nncityp rndSeed,
    nncftyp pBack,
    const bool weights,
    const bool idxWeights,
    const bool idxInput
) {

struct Multi {
    NNNet currNN;
    NNNet bestNN;
    NNNet testNN;
    TVFlt currErrors;
    TVFlt bestErrors;
    nnftyp testError;
};

    FRnd rnd(rndSeed);
    nnityp sumParts = 0;

    for( int i=0 ; i<parts.size() ; i++ ) {
        sumParts += parts[i];
    }
    if( sumParts > data.size() ) {
        abort();
    }
    nncityp testSize = data.size() - sumParts;
    data.shuffle(rnd(),sumParts);
    nnityp sum;
    QVector<Multi> nn( 13 ); // multi
    for( int i=0 ; i<nn.size() ; i++ ) {
        nn[i].bestNN = nn[i].currNN = *this;
        if( testSize > 0 ) {
            nn[i].testNN = *this;
        }
        sum = 0;
        for( int j=0 ; j<parts.size() ; j++ ) {
            nn[i].currErrors.append( nn[i].currNN.error( data , bigPenal , sum , sum + parts[j] ) );
            nn[i].bestErrors.append( nn[i].bestNN.error( data , bigPenal , sum , sum + parts[j] ) );
            sum += parts[j];
        }
        if( sum + testSize != data.size() ) {
            abort();
        }
        if( testSize > 0 ) {
           nn[i].testError = nn[i].testNN.error( data , bigPenal , sum , data.size() );
        }
    }

}


nnftyp NNNet::error( CTVFlt &inp , CTVFlt &out , TVFlt &obuf  ) const {
    nnftyp sum = 0;
    compute( inp , obuf );
    for( nnityp i=0,j=offsetOut() ; i<size_o ; i++,j++ ) {
        nncftyp t = obuf[j] - out[i];
        sum += t * t;
    }
    return sum;
}

nnftyp NNNet::error( CNNRecord &rec , TVFlt &obuf ) const {
    return error( rec.getInps() , rec.getOuts() , obuf );
}

nnftyp NNNet::error( CNNData &data , nncftyp bigpenal, nnityp start , nnityp end ) const {
    nnftyp sum = 0;
    TVFlt obuf( sizeBuf() );
    if( start < 0 ) {
        start = 0;
        end   = data.size();
    }
    #pragma omp parallel for reduction(+:sum) firstprivate(obuf)
    for( nnityp i=start ; i<end ; i++ ) {
        sum += error( data[i].getInps() , data[i].getOuts() , obuf );
    }
    sum /= (end - start) * size_o;
    for( nnityp i=0 ; i<sizeWeights() ; i++ ) {
        sum += bigpenal * weights[i] * weights[i];
    }
    return sum;
}

// Oblicza
nnftyp NNNet::simpClass( CNNData &data ) const {
    TVFlt obuf( sizeBuf() );
    nnftyp sum = 0;
    for( nnityp i=0 ; i<data.size() ; i++ ) {
        compute( data[i].getInps() , obuf );
        if( data[i].getOut(0) == 0 && obuf.last() < 0.5 )
            sum ++;
        if( data[i].getOut(0) == 1 && obuf.last() > 0.5 )
            sum ++;
    }
    return sum / data.size();
}

// Zapisuje do pliku nowe dane wygenerowane siec?? neuronow??.
void NNNet::mkNewData( const QString& path, CNNData &data ) const {
    QFile file( path );
    PEXCP( file.open(QFile::WriteOnly|QFile::Truncate) );
    QTextStream fstr(&file);
    for( nnityp i=0 ; i<data.size() ; i++ ) {
        CTVFlt out = compute( data[i].getInps() );
        for( nnityp j=0 ; j<data.sizeInp() ; j++ )
            fstr << data[i].getInp(j) << ",";
        for( nnityp j=0 ; j<out.size() ; j++ )
            fstr << out[j] << ",";
        for( nnityp j=0 ; j<out.size() ; j++ ) {
            if( j!=0 )
                fstr << ",";
            fstr << data[i].getOut(j);
        }
        fstr << endl;
    }
    file.close();
}


// Wczytuje nast??pn?? lini?? ze strumienia wej??ciowego.
// Opuszcza komentarze (za znakiem #) i puste linie.
static QString nextLine( QTextStream &inp ) {
    QString line;
    do {
        PEXCP( !inp.atEnd() );
        line = inp.readLine();
        int coment = line.indexOf('#');
        if( coment != -1 ) {
            line = line.left(coment);
        }
        line = line.trimmed().toLower();
    } while( line.isEmpty() );
    return line;
}

// Wczytuje wektor int??w za etykiet?? i dwukropkiem.
// Wykonuje asercje, sprawdza czy ilo???? int??w mie??ci si?? w
// przedziale od min do max.
static TVInt getInts( const QString &label, const QString& line, nncityp min=-1, nncityp max=-1) {
    QRegExp rex( QString("^\\s*%1\\s*:\\s*([0-9\\+\\-\\,\\s]+)$").arg( QRegExp::escape(label) ) );
    rex.indexIn(line);
    PEXCP( rex.captureCount() == 1 );
    const QStringList ints = rex.cap(1).split(QRegExp("[\\s,;]+"),QString::SkipEmptyParts);
    if( !(min==-1 || ints.size()>=min) )
        qDebug() << line << min << max;
    PEXCP( min==-1 || ints.size()>=min);
    PEXCP( max==-1 || ints.size()<=max);
    TVInt out;
    for( int i=0 ; i<ints.size() ; i++ ) {
        bool ok;
        out.append( ints[i].toInt(&ok) );
        PEXCP(ok);
    }
    return out;
}

static TVInt getInts( const QString &label, QTextStream &finp, nncityp min=-1, nncityp max=-1) {
    return getInts( label , nextLine(finp), min , max );
}

// Wczytuje jednego inta za etykiet??.
static nnityp getInt( const QString &label, const QString &line ) {
    CTVInt v = getInts( label ,line, 1, 1 );
    PEXCP( v.size() == 1 );
    return v[0];
}


// Wczytuje jednego inta za etykiet??.
static nnityp getInt( const QString &label, QTextStream &inp ) {
    return getInt( label , nextLine(inp) );
}


static TVFlt getFloats( const QString &label, const QString& line, nncityp min=-1, nncityp max=-1) {
    QRegExp rex( QString("^\\s*%1\\s*:\\s*([0-9\\.\\+\\-\\,\\sEe]*)$").arg( QRegExp::escape(label) ) );
    rex.indexIn(line);
    PEXCP( rex.captureCount() == 1 );
    const QStringList floats = rex.cap(1).split( QRegExp("[\\s,;]+") , QString::SkipEmptyParts );
    PEXCP( min==-1 || floats.size()>=min);
    PEXCP( max==-1 || floats.size()<=max);
    TVFlt out;
    for( int i=0 ; i<floats.size() ; i++ ) {
        bool ok;
        out.append( floats[i].toDouble(&ok) );
        PEXCP(ok);
    }
    return out;
}

// Wczytuje jednego floata za etykiet??.
static nnftyp getFlt( const QString &label, QTextStream &inp ) {
    QString line = nextLine(inp);
    CTVFlt v = getFloats( label ,line, 1, 1 );
    PEXCP(v.size()==1);
    return v[0];
}

//Zwraca napisy za etykiet??.
static TVStr getStrings( const QString &label, const QString& line, nncityp min=-1, nncityp max=-1) {
    QRegExp rex( QString("^\\s*%1\\s*:\\s*(.*)$").arg( QRegExp::escape(label) ) );
    rex.indexIn(line);
    PEXCP( rex.captureCount() == 1 );
    CTVStr strs = rex.cap(1).split( QRegExp("[\\s,;]+") , QString::SkipEmptyParts ).toVector();
    PEXCP( min==-1 || strs.size()>=min );
    PEXCP( max==-1 || strs.size()<=max );
    return strs;
}

// Zwraca napis za etykiet??.
static QString getString( const QString &label, const QString& line ) {
    return getStrings( label , line , 1 , 1 )[0];
}

// Lista funkcji aktywacji.
const static QStringList strActvs{"lin", "mlin", "suni", "sbip", "sbip_l", "relu", "nname"};

// Zwraca funkcj?? aktywacji neuronu
static NNActv getActv( QTextStream &finp ) {
    const QString actv = getString( "actv", nextLine(finp) );
    PEXCP( strActvs.indexOf(actv) != -1 );
    return (NNActv) strActvs.indexOf(actv);
}


// Odczytanie sieci neuronowej z pliku
void NNNet::read(const QString& path) {
    QFile file(path);
    PEXCP(file.open(QFile::ReadOnly));
    QTextStream finp(&file);

    bool ok;
    TVFlt tmp_f;
    TVInt tmp_i;

    size_i  = getInt( "size_inputs" , finp );                  // Wczytuje ilo???? wej???? sieci.
    size_o  = getInt( "size_outputs" , finp  );                // Wczytuje ilo???? wyj???? sieci.
    nncityp size_neurons = getInt( "size_neurons" , finp  );   // Wczytuje ilo???? neuron??w.
    nncityp size_weights = getInt( "size_weights" , finp  );   // Wczytuje ilo???? wag.
    nncityp size_convols = getInt( "size_convols" , finp  );   // Wczytuje ilo???? splot??w.

    weights_penal = getFlt( "weights_penal" , finp  ); // Wczytuje kar?? za du??e wagi.
    signal_penal = getFlt( "signal_penal" , finp  );   // Wczytuje kar?? za du??e warto??ci wej??cia.

    rnd_seed = getInt( "rnd_seed" , finp  ); // Wczytuje zarodek liczb losowych do inicjacji wag dla kt??rych nie podano warto??ci pocz??tkowej.
    FRnd rnd(rnd_seed);

    tmp_f = getFloats( "weights" , nextLine(finp) , 2 , 2 ); // Wczytuje domy??lny przedzia?? wag, b??dzie u??yte gdy nie podano innej warto??ci dla wagi.
    gmin_w = tmp_f[0];
    gmax_w = tmp_f[1];
    PEXCP( gmin_w <= gmax_w );

    weights.resize(size_weights);      // Pami???? na wagi.
    min_w.fill(gmin_w,size_weights);    // Pami???? na minimalne ograniczenia wag, inicjujemy min_w.
    max_w.fill(gmax_w,size_weights);    // Pami???? na maksymalne ograniczenia wag, inicjujemy max_w.

    for( nnityp i=0 ; i<size_weights ; i++ ) {     // Po wszystkich wagach.
        weights[i] = rnd.getF(gmin_w,gmax_w);    // Losujemy wagi z domy??lnego przedzia??u.
    }

    QString line;

    while( (line = nextLine(finp)) != "endweights" ) { // Wczytuj definicje wag a?? do napotkania endWeight.
        QRegExp rxp("^\\s*(weight_([0-9]+))");         // Regexp do wczytania etykiety waga_numer.
        PEXCP( rxp.indexIn(line) == 0 );
        PEXCP( rxp.captureCount() == 2 );
        nncityp nr = rxp.cap(2).toUInt(&ok);           // Pobierz numer wagi.

        PEXCP(ok);
        PEXCP( nr>=0 && nr < size_weights );
        tmp_f = getFloats( rxp.cap(1) , line , 0 , 3 );
        if( tmp_f.size() >= 1 ) weights[nr] = tmp_f[0];
        if( tmp_f.size() >= 2 ) min_w[nr]   = tmp_f[1];
        if( tmp_f.size() >= 3 ) max_w[nr]   = tmp_f[2];
        PEXCP( min_w[nr] <= max_w[nr] );
        if( weights[nr] > max_w[nr] )
            weights[nr] = max_w[nr];
        if( weights[nr] < min_w[nr] )
            weights[nr] = min_w[nr];
    }

    convs.clear();
    convs.resize( size_convols );

    neurons.clear();
    neurons.reserve( size_neurons );
    TVInt cntn(size_neurons,0);

    while( (line = nextLine(finp)) != "endneurons" ) {
        nncityp nr = getInt( "neuron" , line );
        PEXCP( nr == neurons.size() );
        PEXCP( cntn[nr] == 0 );
        cntn[nr] = 1;

        NNNeuron neuron;

        // funkcja aktywacji
        neuron.actv = getActv(finp);

        // splot
        neuron.conv = getInt( "conv" , finp );

        // wej??cia neuronu.
        neuron.idx_i = getInts("inp",finp,1);
        for( int i=0 ; i<neuron.idx_i.size() ; i++ ) {
            PEXCP( neuron.idx_i[i] >= 0 );
            PEXCP( neuron.idx_i[i] < nr + size_i );
        }

        // zakres indeks??w wej???? neuronu
        tmp_i = getInts("rng_inputs",finp,2,2);
        neuron.min_i = tmp_i[0];
        neuron.max_i = tmp_i[1];
        PEXCP( neuron.max_i - neuron.min_i + 1 >= neuron.idx_i.size() ); // wej??cia musz?? by?? r????ne, wi??c przedzia?? musi by?? wi??kszy lub r??wny ni?? ilo???? wej????

        // indeksy wag
        neuron.idx_w = getInts("weights", finp,neuron.idx_i.size(), neuron.idx_i.size() );
        for( int i=0 ; i<neuron.idx_w.size() ; i++ ) {
            PEXCP( neuron.idx_w[i] >= 0 );
            PEXCP( neuron.idx_w[i] < size_weights );
        }

        // zakres wag
        tmp_i = getInts("rng_weights",finp,2,2);
        neuron.min_w = tmp_i[0];
        neuron.max_w = tmp_i[1];
        PEXCP( neuron.max_w >= neuron.min_w );

        tmp_i = getInts("rng_size",finp,2,2);
        neuron.min_s = tmp_i[0];
        neuron.max_s = tmp_i[1];
        PEXCP( neuron.min_s <= neuron.max_s );
        PEXCP( neuron.idx_i.size() <= neuron.max_s );
        PEXCP( neuron.idx_i.size() >= neuron.min_s );

        if( neuron.conv >= 0 ) {
            TVInt &conv = convs[ neuron.conv ];
            for( int i=0 ; i<conv.size() ; i++ ) {
                CTVInt &idx_w =  neurons[ conv[i] ].idx_w ;
                PEXCP( idx_w.size() == neuron.idx_w.size() );
                for( int i=0 ; i<idx_w.size() ; i++ ) {
                    PEXCP( idx_w[i] == neuron.idx_w[i] );
                }
                PEXCP( neurons[ conv[i] ].min_s == neuron.min_s );
                PEXCP( neurons[ conv[i] ].max_s == neuron.max_s );
                PEXCP( neurons[ conv[i] ].min_w == neuron.min_w );
                PEXCP( neurons[ conv[i] ].max_w == neuron.max_w );
            }
            conv.append( nr );
        }

        neurons.append( neuron );
    }
    neurons.squeeze();
    file.close();
}


void NNNet::save(const QString& path) const {
    QFile file(path);
    PEXCP( file.open(QFile::WriteOnly|QFile::Truncate) );
    QTextStream fout(&file);

    fout.setCodec("utf-8");
    fout.setFieldWidth(8);
    fout.setRealNumberPrecision(15);
    fout.setRealNumberNotation(QTextStream::FixedNotation);
    fout.setFieldAlignment(QTextStream::AlignRight);

    fout << "size_inputs:  "  << size_i         << QString::fromUtf8(" # rozmiar wej??cia") << endl;
    fout << "size_outputs: "  << size_o         << QString::fromUtf8(" # rozmiar wyj??cia") << endl;
    fout << "size_neurons: "  << neurons.size() << QString::fromUtf8(" # ilo???? neuron??w")  << endl;
    fout << "size_weights: "  << weights.size() << QString::fromUtf8(" # ilo???? wag")       << endl;
    fout << "size_convols: "  << convs.size()   << QString::fromUtf8(" # ilo???? splot??w")   << endl;
    fout << endl;


    fout << "weights_penal: "  << forcesign << weights_penal << QString::fromUtf8(" # kara za du??e wagi")    << endl;
    fout << "signal_penal:  "  << forcesign << signal_penal  << QString::fromUtf8(" # kara za du??e sygna??y") << endl;
    fout << endl;


    fout << "rnd_seed: "      << rnd_seed  << QString::fromUtf8(" # zarodek liczb losowych") << endl;
    fout << endl;


    fout << qSetFieldWidth(0) << "weights: " << forcesign << qSetFieldWidth(20) << gmin_w  << qSetFieldWidth(0) << ", "  << qSetFieldWidth(20) << forcesign << gmax_w << QString::fromUtf8(" # wagi") << endl;
    for( nnityp i=0 ; i<weights.size() ; i++ ) {
        fout << noforcesign << qSetFieldWidth(0);
        fout << "  weight_" << i << ": ";
        fout << forcesign;
        fout << qSetFieldWidth(20) << weights[i] << qSetFieldWidth(0) << ", ";
        fout << qSetFieldWidth(20) << min_w[i]   << qSetFieldWidth(0) << ", ";
        fout << qSetFieldWidth(20) << max_w[i]   << qSetFieldWidth(0) << ", ";
        fout << "# waga nr " << noforcesign << (i+1) << endl;
    }
    fout << "endWeights" << endl << endl << endl;

    for( nnityp i=0 ; i<neurons.size() ; i++ ) {
        fout << noforcesign << qSetFieldWidth(0) << "neuron:" << i << "      #neuron " << i << "; input: " << (i+size_i) << endl;
        fout << "  actv: " <<  strActvs[ neurons[i].actv ] << endl;
        fout << "  conv: " << neurons[i].conv << endl;

        fout << "  inp: ";
        for( nnityp j=0 ; j<neurons[i].idx_i.size() ; j++ ) {
            if( j>0 )
                fout << ", ";
            fout << neurons[i].idx_i[j];
        }
        fout << endl;

        fout << "  rng_inputs: " << neurons[i].min_i << ", " << neurons[i].max_i << endl;

        fout << "  weights: ";
        for( nnityp j=0 ; j<neurons[i].idx_w.size() ; j++ ) {
            if( j>0 )
                fout << ", ";
            fout << neurons[i].idx_w[j];
        }
        fout << endl;

        fout << "  rng_weights: " << neurons[i].min_w << ", " << neurons[i].max_w << endl;
        fout << "  rng_size: "    << neurons[i].min_s << ", " << neurons[i].max_s << endl;

        fout << endl;
    }

    fout << "endNeurons" << endl;

    file.close();
}

// Zaburza warto???? losowej wagi.
bool NNNet::chaosWeight( FRnd &rnd, cftyp strength ) {
    TVInt v;                                      // Indeksy wag kt??re mo??na zaburzy??.
    for( nnityp i=0 ; i<weights.size() ; i++ ) {  // Po wszystkich wagach.
        if( max_w[i] - min_w[i] > 0.000001 )      // Je??li jest jaki?? sensowny przedzia??, to
            v.append(i);                          // wag?? mo??na modyfikowa??.
    }
    if( v.size() == 0 )                           // Je??li ??adna waga nie nadaje si?? do modyfikacji, to
        return false;                             // Nie uda??o si?? zaburzy?? ??adnej wagi.
    nncityp r = v[ rnd() % v.size() ];            // Wylosuj wag??.
    rnd.chaos( weights[r] , min_w[r] , max_w[r] , strength ); // Zaburz wag??.
    return true;                                  // Uda??o si?? zaburzy?? wag??.
}

// Zaburza warto???? wszystkich wag
void NNNet::chaosWeights(nncityp rndSeed, cftyp strength , nncityp loops) {
    FRnd rnd(rndSeed);
    for( nnityp loop=0 ; loop<loops ; loop++ ) {
        for( nnityp i=0 ; i<weights.size() ; i++ ) {             // Po wszystkich wagach.
            weights[i] += rnd.getF( -strength , +strength );
        }
    }
    for( nnityp i=0 ; i<weights.size() ; i++ ) {                 // Po wszystkich wagach.
        if( weights[i] < min_w[i] ) {
            weights[i] = min_w[i];
        }
        if( weights[i] > max_w[i] ) {
            weights[i] = max_w[i];
        }
    }
}


// Zaburza losowo indeks wagi w losowo wybranym neuronie.
bool NNNet::chaosIdxWeight( FRnd &rnd ) {
    NNNeuron &n = neurons[ rnd() % neurons.size() ];  // Wylosuj neuron.
    nncityp idxw = rnd.getI( n.min_w , n.max_w );     // Wylosuj indeks wagi z dozwolonego przedzia??u wag.
    nncityp pos = rnd() % n.idx_w.size();             // Wylosuj pozycj?? na kt??r?? wstawi?? nowy indeks.
    n.idx_w[pos] = idxw;                              // Wstaw nowy indeks.
    if( n.isConvolution() ) {                         // Je??li neuron jest w splocie z innymi neuronami.
        CTVInt &conv = convs[n.conv];                 // Wektor neuron??w splecionych.
        for( nnityp i=0 ; i<conv.size() ; i++ ) {     // We wszystkich neuron??w ze splotu
            neurons[ conv[i] ].idx_w[pos] = idxw;     // na identycznej pozycji zmie?? identyczny indeks.
        }
    }
    return true;                                      // Zwr???? informacj?? ??e udalo si?? zaburzy?? indeks wagi w neuronie.
}

// Zaburza losowo indeks wej??cia neuronu. Zak??adamy ??e
// wej??cia przed zaburzeniem i po zaburzeniu s?? unikalne,
// nie powtarzaj?? si??.
bool NNNet::chaosIdxInput( FRnd &rnd ) {
    TVInt v;                                           // Miejsce na r????ne indeksy.
    for( nnityp i=0 ; i<neurons.size() ; i++ ) {       // Po wszystkich neuronach.
        CNNNeuron &n = neurons[i];                     // Tylko skr??t do neuronu.
        if( n.idx_i.size() < n.max_i - n.min_i + 1 )   // Je??li ilo???? wej???? jest mniejsza ni?? maksymalna ilo???? wej????, to
            v.append(i);                               // jest mo??liwo???? zaburzenia z utrzymaniem unikalno??ci.
    }
    if( v.size() == 0 )                                // Je??li ??adnego neuronu nie mo??na zaburzy??, to
        return false;                                  // zwr???? informacj??, ??e nie uda??o si?? zaburzenie.
    NNNeuron &n = neurons[ v[ rnd() % v.size() ] ];    // Wylosuj neuron z neuron??w nadaj??cych si?? do modyfikacji.
    QSet<nnityp> idx_i;                                // Zbi??r na istniej??ce wej??cia neuronu.
    for( nnityp i=0 ; i<n.idx_i.size() ; i++ ) {       // Dodaj do zbioru
        idx_i.insert( n.idx_i[i] );                    // wszystkie istniej??ce wej??cia.
    }
    v.clear();                                         // Teraz v przyda si?? na wej??cia kt??rych neuron nie ma.
    for( nnityp i=n.min_i ; i<=n.max_i ; i++ ) {       // Iteruj po wej??ciach jakie neuron mo??e mie??.
        if( ! idx_i.contains(i) )                      // Je??li neuron nie ma wej??cia, to
            v.append(i);                               // dodaj to wej??cie.
    }
    nncityp inp = v[ rnd() % v.size() ];               // Wylosuj nowe wej??cie dla neuronu.
    nncityp pos = rnd() % n.idx_i.size();              // Wylosuj pozycj?? na kt??rej stare wej??cie zostanie usuni??te.
    EXCP( n.idx_i[pos] != inp );                       // Nie dublujemy wej????, uproszczona asercja.
    n.idx_i[pos] = inp;                                // Zast??p stare wej??cie nowym, losowym.
    return true;                                       // Uda??o si?? zaburzenie.
}

// Zaburzenie losowe w sieci neuronowej.
bool NNNet::chaos(
    FRnd &rnd ,     // generator liczb losowych
    cftyp strength ,  // si??a zaburzenia w przypadku zaburzania warto??ci (nie indeksu) wagi.
    const bool weights,
    const bool idx_weights,
    const bool idx_input
) {
    for( nnityp i=0 ; i<1000 ; i++ ) {                 // Pr??buj 1000 razy wykona?? jakie?? zaburzenie losowe.
        TVInt v{0,1,2};                                // S?? trzy mo??liwosci zaburze?? losowych.
        while( v.size() > 0 ) {                        // Do p??ki nie wykorzystano wszystkich mo??liwo??ci.
            nncityp r = rnd() % v.size();              // Wylosuj mo??liwo????, a w??a??ciwie to indeks mo??liwo??ci.
            switch( v[ r ] ) {                         // Skok do procedury zaburzenia
                case 0:
                    if( weights && chaosWeight(rnd,strength) )    // Jak si?? uda??o zaburzenie, to zwro?? informacj??, ??e si?? uda??o.
                        return true;
                    break;
                case 1:
                    if( idx_weights &&  chaosIdxWeight(rnd) )           // Jak wy??ej.
                        return true;
                    break;
                case 2:
                    if( idx_input && chaosIdxInput(rnd) )             // Jak wy??ej.
                        return true;
                    break;
            }
            v.remove(r);                                 // Usu?? mo??liwo???? spod indeksu r i spr??buj nast??pnej mo??liwo??ci.
        }
    }
    return false;                                        // Nie uda??o si?? zaburzy?? pomimo 1000 pr??b.
}

}
