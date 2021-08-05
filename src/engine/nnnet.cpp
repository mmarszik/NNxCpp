#include <omp.h>
#include <QFile>
#include <QTextStream>
#include <QRegExp>
#include <cmath>
#include <QtDebug>
#include "nnnet.h"
#include "src/misc/critical.h"

namespace NsNet {

// Funkcja aktywacji
static nnftyp factivate( nncftyp inp, NNActv actv ) {
    switch( actv ) {
        case NNA_LIN:   return inp;
        case NNA_SUNI:  return 1.0 / ( 1.0 + exp(-inp) );
        case NNA_SBIP:  return 2.0 / ( 1.0 + exp(-inp) ) - 1.0;
        case NNA_RELU:  return log( 1.0 + exp(inp) );
        case NNA_NNAME: return 1.0 / ( 1.0 + fabs(inp) );
    }
    return 0;
}

// Funkcja pochodna
static nnftyp fderivate( nncftyp out, NNActv actv ) {
    nnftyp tmp;
    switch( actv ) {
        case NNA_LIN:   return 1;
        case NNA_SUNI:  return out * (1.0 - out);
        case NNA_SBIP:  return 0.5 * (1.0 - out*out);
        case NNA_RELU:  return 1.0 / ( exp(-out) + 1 );
        case NNA_NNAME: tmp = fabs( out / ( 1 - fabs(out) ) ) + 1; return 1.0 / ( tmp * tmp );
    }
    return 0;
}

/*
void NNNet::compute( CTVFlt &inp , TVFlt &out , TVFlt &obuf ) const {
    EXCP( inp.size() == size_i );         // Sieć jest przystosowana do jednego rozmiaru wejścia i
    EXCP( out.size() == size_o );         // wyjścia.

    for( nnityp i=0 ; i<size_i ; i++ ) {  // Skopiuj wejścia.
        obuf[i] = inp[i];
    }

    // obliczenia
    for( nnityp i=0 ; i<neurons.size() ; i++ ) {  // Po wszystkich neuronach.
        CNNNeuron &neuron = neurons[i];           // Skrót do bieżącego neuronu.
        nnftyp nout = 0;                          // Pamięć na wyjście neuronu inicjuj zerem.
        CTVInt &idx_i = neuron.idx_i;             // Skrót do indeksów wejść.
        CTVInt &idx_w = neuron.idx_w;             // Skrót do indeksów wag.
        EXCP( idx_i.size() == idx_w.size() );     // Ilość wag i wejść musi być taka sama.
        for( nnityp j=0 ; j<idx_i.size() ; j++ ) {
            if( idx_i[j] >= i + size_i ) abort();
            nout += weights[ idx_w[j] ] * obuf[ idx_i[j] ];  // Suma ważona.
        }
        obuf[ i + size_i ] = factivate( nout , neuron.actv ); // Wyjście neuronu, potem zoptymalizować.
    }

    for( nnityp i=0 ; i<size_o ; i++ ) {       // Po wszystkich wyjściach.
        out[i] = obuf[obuf.size()-size_o+i]; // Na końcu jest odpowiedź sieci, skopiuj do wyjścia.
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
    TVFlt obuf( size_i + neurons.size() ); // Pamięć na wyjścia neuronów.
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
    TVFlt obuf( size_i + neurons.size() ); // Pamięć na wyjścia neuronów.
    return error( inp , out , obuf );
}

nnftyp NNNet::error( CNNRecord &rec , TVFlt &obuf ) const {
    return error( rec.getInps() , rec.getOuts() , obuf );
}

nnftyp NNNet::error( CNNRecord &rec ) const {
    TVFlt obuf( size_i + neurons.size() ); // Pamięć na wyjścia neuronów.
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
    EXCP( inp.size() == size_i );                     // Sieć jest przystosowana do jednego rozmiaru wejścia i
    EXCP( obuf.size() == size_o + neurons.size() );   // wyjścia.
    for( nnityp i=0 ; i<size_i ; i++ ) {              // Skopiuj wejścia.
        obuf[i] = inp[i];
    }
    // obliczenia
    for( nnityp i=0 ; i<neurons.size() ; i++ ) {  // Po wszystkich neuronach.
        CNNNeuron &neuron = neurons[i];           // Skrót do bieżącego neuronu.
        nnftyp nout = 0;                          // Pamięć na wyjście neuronu inicjuj zerem.
        CTVInt &idx_i = neuron.idx_i;             // Skrót do indeksów wejść.
        CTVInt &idx_w = neuron.idx_w;             // Skrót do indeksów wag.
        EXCP( idx_i.size() == idx_w.size() );     // Ilość wag i wejść musi być taka sama.
        for( nnityp j=0 ; j<idx_i.size() ; j++ ) {
//            qDebug() << idx_w[j] << weights[ idx_w[j] ];
            nout += weights[ idx_w[j] ] * obuf[ idx_i[j] ];  // Suma ważona.
        }
        obuf[ i + size_i ] = factivate( nout , neuron.actv ); // Wyjście neuronu, potem zoptymalizować.
    }
}


TVFlt NNNet::compute( CTVFlt &inp ) const {
    TVFlt obuf(suzeBuf());
    compute( inp , obuf );
    return obuf.mid( offsetOut() );
}


void NNNet::gradient( CTVFlt &inp , CTVFlt &out , TVFlt &obuf , TVFlt &ibuf , TVFlt &grad  ) const {
    QTextStream stdo(stdout);

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


TVFlt NNNet::gradient( CNNData &data , nncftyp bigpenal ) const {
    nncityp mt = omp_get_max_threads();

    NNVec< TVFlt > grads( mt );
    grads.fill( TVFlt( sizeWeights() , 0 ) );

    TVFlt obufs( suzeBuf() );
    TVFlt ibufs( suzeBuf() );

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

    // f = w1^2 + w2^2 + ... + wn^2
    // df/dw1 = 2( w1^2 + w2^2 + ... wn^2)

    if( bigpenal > 0 ) {
        for( nnityp i=0 ; i<sizeWeights() ; i++ )
            grad[i] += 2 * weights[i] * bigpenal;
    }

    return grad;
}

// Liczy gradient tylko dla wskazanych numerów wag
TVFlt NNNet::subGradient( CNNData &data , CTVInt &sub ) const {
    CTVFlt gr1 = gradient( data );
    TVFlt gr2( sub.size() );
    for( nnityp i=0 ; i<sub.size() ; i++ )
        gr2[i] = gr1[sub[i]];
    return gr2;
}

static nnftyp fnorm( CTVFlt &v ) {
    nnftyp norm = 0;
    for( nnityp i=0 ; i<v.size() ; i++ )
        norm += v[i] * v[i];
    return sqrt(norm);
}

static nnftyp avgNorm( CTVFlt &v ) {
    nnftyp norm = 0;
    for( nnityp i=0 ; i<v.size() ; i++ )
        norm += v[i] * v[i];
    return sqrt( norm / v.size() );
}

static void mkNorm( TVFlt &v ) {
    nncftyp n = fnorm(v);
    for( nnityp i=0 ; i<v.size() ; i++ )
        v[i] /= n;
}

static TVFlt mkNorm2( CTVFlt &v ) {
    nncftyp n = fnorm(v);
    TVFlt out( v.size() );
    for( nnityp i=0 ; i<v.size() ; i++ )
        out[i] = v[i] / n;
    return out;
}


static TVFlt& toLearn( TVFlt &gr , CTVFlt &tolearn ) {
    for( nnityp i=0 ; i<std::min(gr.size(),tolearn.size()) ; i++ )
        gr[i] *= tolearn[i];
    return gr;
}

void NNNet::rawDescent( CNNData &data , nncityp loops , nncftyp step , CTVFlt &tolearn ) {
    for( nnityp loop = 0 ; loop < loops ; loop ++ ) {
        TVFlt gr = gradient( data );
        toLearn( gr , tolearn );
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

void NNNet::rawMomentum( CNNData &data , nncityp loops , nncftyp step , TVFlt &p , nncftyp mom,  nncftyp bigpenal, CTVFlt &tolearn) {
    for( nnityp loop = 0 ; loop < loops ; loop++ ) {
        TVFlt g = gradient(data,bigpenal);
        toLearn( g , tolearn );
        mkNorm( g );
        for( nnityp i=0 ; i<p.size() ; i++ )
            p[i] = p[i] * mom + g[i] * (1.0-mom);
        for( nnityp i=0 ; i<p.size() ; i++ ) {
            weights[i] -= p[i] * step;
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
        stdOut << "   best test";
    }
    stdOut<< "      step  ||gradient||   ||weights||         ||p||  fails     time" << endl;
    nnityp loop;
    const time_t start = time(NULL);
    time_t currTime = start;
    time_t lastShow = currTime;
    time_t showTime = 1;
    nnftyp er3 = 0;
    for( loop = 1 ; (currTime-start) <= maxTime && step >= minStep && (maxFailsTest==0 || failsTest < maxFailsTest); loop++ ) {
        currTime = time(NULL);
        rawMomentum( learn , subLoops , step , p , mom , bigPenal , toLearn );
        nncftyp er2 = error( learn , bigPenal );

        if( er2 > er1 ) {
            weights = bestWeights;
            step *= 0.5;
            success = 0;
            fails ++ ;
        } else {
            bestWeights = weights;

            if( er2 < er1 ) {
                er1 = er2;
                fails = 0;
                if( ++success >= 1 ) {
                    step *= 1.20;
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
            stdOut << qSetFieldWidth(9) << qSetRealNumberPrecision(6) << step;
            stdOut << qSetFieldWidth(0) << " ";
            stdOut << qSetFieldWidth(13)<< avgNorm( gradient(learn) );
            stdOut << qSetFieldWidth(0) << " ";
            stdOut << qSetFieldWidth(13) << avgNorm(weights);
            stdOut << qSetFieldWidth(0) << " ";
            stdOut << qSetFieldWidth(13) << fnorm(p);
            stdOut << qSetFieldWidth(0) << " ";
            stdOut << qSetFieldWidth(6) << fails;
            stdOut << qSetFieldWidth(0) << " ";
            stdOut << qSetFieldWidth(7) << (currTime-start);
            stdOut << qSetFieldWidth(0) << "s";
            stdOut << endl;

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
    weights = bestWeights;

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
    nncityp size = sub.size();                   // Skrót do rozmiaru pod zbioru optymalizowanych wag.
    TVFlt g = subGradient( data , sub );         // Gradient
    nnftyp ng = fnorm(g);                        // Norma gradientu.
    TVFlt pg(size);                              // Poprzedni gradient
    TVFlt s(size), r(size), rv(size), vr(size);  // Wektory pomocnicze.
    TVFlt hes( size * size );                    // Hesjan.
    nnftyp er1 = error( data );                  // Błąd.
    TVFlt best = weights;                        // Kopia najlepszych wag.
    nnityp fail  =  0;                           // Ilość nieudanych kroków.
    nnityp reset = 10;                           // Pewna odległość od resetu
    bfgsResetHes( hes );                         // Reset hesjanu.
    for( nnityp loop=0 ; loop<loops && ng > 1E-6 ; loop++ ) {
        CTVFlt prev = weights;                   // kopia wag
        CTVFlt p = bfgsDir( hes , g );           // znormalizowany kierunek zgodny z bfgs
        minDir( weights , p , sub , step / reset ); // Zanim hesjan dobrze się nie uaktualni, zmniejszamy krok.
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
        if( fabs(max-min) > 0.000001 ) {
            weights[i] = rnd.getF( min , max );
        } else {
            weights[i] = rnd.getF( min_w[i] , max_w[i] );
        }
    }
}

int NNNet::forceIdx( CNNData &data , nncftyp bigPenal, nncityp loops, const bool show ) {
    int increse = 0;
    QTextStream stdo(stdout);
    stdo.setRealNumberPrecision(8);
    stdo.setRealNumberNotation( QTextStream::FixedNotation );
    nnftyp er = error(data);
    for( nnityp loop=0 ; loop<loops ; loop ++ ) {
        bool work = false;
        for( nnityp i=0 ; i<neurons.size() ; i++ ) {
            NNNeuron &n = neurons[i];
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

    for( nnityp loop=0 ; fails < maxFails && ( maxTime == 0 || (currTime - start) <= maxTime ) ; loop ++ ) {
        data.shuffle(rnd);
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

void NNNet::toUniqueWeights() {
    TVFlt newWeights;
    TVFlt newMin;
    TVFlt newMax;
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


//MM: Uczenie poprzez losową zmianę idnexów wag (a nie samych wag).
void NNNet::learnRndIdx(
    CNNData &data ,            //MM: dane uczące
    const time_t maxTime ,     //MM: maksymalna ilość czasu
    nncityp maxNotLearn ,      //MM: maksymalna ilość iteracji bez minimalnego spadku błędu
    nncftyp minError ,         //MM: wartość minimalnego spadku błędu
    nncftyp bigpenal ,         //MM: kara za duże wagi
    nncityp rndSeed ,          //MM: zarodek liczb losowych
    nncftyp pBack              //MM: prawdopodobieństwo powrotu do lepszego rozwiązania
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
    nnityp loop;
    const time_t start = time(NULL);
    time_t currTime = start;
    for( loop=0 ; currTime - start <= maxTime && (maxnoti==0 || noti < maxnoti) ; loop++ ) {
        chaos( rnd , strenght , weights, idx_weights, idx_input );
        nncftyp tmp = error( data , bigPenal  );
        if( tmp <= er ) {
            if( tmp < er - mindesc ) {
                stdo << qSetFieldWidth(8)  << loop;
                stdo << qSetFieldWidth(0)  << "] ";
                stdo << qSetFieldWidth(10) << tmp;
                stdo << qSetFieldWidth(0)  << " " << qSetFieldWidth(7) << noti;
                stdo << qSetFieldWidth(0)  << " " << qSetFieldWidth(10) << strenght;
                stdo << qSetFieldWidth(0)  << " " << qSetFieldWidth(0) << (currTime-start);
                stdo << qSetFieldWidth(0)  << "s";
                stdo << endl;
                noti = 0;
            }
            er = tmp;
            best = *this;
        } else if( rnd.getF() < 0.5 ) {
            *this = best;
        }
        if( (loop & 0x0) == 0 ) {
            const time_t tmp = time(NULL);
            while( currTime < tmp) {
                strenght *= decay;
                currTime++;
            }
        }
        noti ++ ;
    }
    *this = best;
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
    data.shuffle(rnd,sumParts);
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
    TVFlt obuf( suzeBuf() );
    if( start < 0 ) {
        start = 0;
        end   = data.size();
    }
    #pragma omp parallel for reduction(+:sum) firstprivate(obuf) schedule(static)
    for( nnityp i=start ; i<end ; i++ ) {
        sum += error( data[i].getInps() , data[i].getOuts() , obuf );
    }
    sum /= (end - start) * size_o;
    for( nnityp i=0 ; i<sizeWeights() ; i++ )
        sum += bigpenal * weights[i] * weights[i];
    return sum;
}

// Oblicza
nnftyp NNNet::simpClass( CNNData &data ) const {
    TVFlt obuf( suzeBuf() );
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

// Zapisuje do pliku nowe dane wygenerowane siecą neuronową.
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


// Wczytuje następną linię ze strumienia wejściowego.
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

// Wczytuje wektor intów za etykietą i dwukropkiem.
// Wykonuje asercje, sprawdza czy ilość intów mieści się w
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

// Wczytuje jednego inta za etykietą.
static nnityp getInt( const QString &label, const QString &line ) {
    CTVInt v = getInts( label ,line, 1, 1 );
    PEXCP( v.size() == 1 );
    return v[0];
}


// Wczytuje jednego inta za etykietą.
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

// Wczytuje jednego floata za etykietą.
static nnftyp getFlt( const QString &label, QTextStream &inp ) {
    QString line = nextLine(inp);
    CTVFlt v = getFloats( label ,line, 1, 1 );
    PEXCP(v.size()==1);
    return v[0];
}

//Zwraca napisy za etykietą.
static TVStr getStrings( const QString &label, const QString& line, nncityp min=-1, nncityp max=-1) {
    QRegExp rex( QString("^\\s*%1\\s*:\\s*(.*)$").arg( QRegExp::escape(label) ) );
    rex.indexIn(line);
    PEXCP( rex.captureCount() == 1 );
    CTVStr strs = rex.cap(1).split( QRegExp("[\\s,;]+") , QString::SkipEmptyParts ).toVector();
    PEXCP( min==-1 || strs.size()>=min );
    PEXCP( max==-1 || strs.size()<=max );
    return strs;
}

// Zwraca napis za etykietą.
static QString getString( const QString &label, const QString& line ) {
    return getStrings( label , line , 1 , 1 )[0];
}

// Lista funkcji aktywacji.
const static QStringList strActvs{"lin", "suni", "sbin", "relu", "nname"};

// Zwraca funkcję aktywacji neuronu
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

    size_i  = getInt( "size_inputs" , finp );                  // Wczytuje ilość wejść sieci.
    size_o  = getInt( "size_outputs" , finp  );                // Wczytuje ilość wyjść sieci.
    nncityp size_neurons = getInt( "size_neurons" , finp  );   // Wczytuje ilość neuronów.
    nncityp size_weights = getInt( "size_weights" , finp  );   // Wczytuje ilość wag.
    nncityp size_convols = getInt( "size_convols" , finp  );   // Wczytuje ilość splotów.

    weights_penal = getFlt( "weights_penal" , finp  ); // Wczytuje karę za duże wagi.
    signal_penal = getFlt( "signal_penal" , finp  );   // Wczytuje karę za duże wartości wejścia.

    rnd_seed = getInt( "rnd_seed" , finp  ); // Wczytuje zarodek liczb losowych do inicjacji wag dla których nie podano wartości początkowej.
    FRnd rnd(rnd_seed);

    tmp_f = getFloats( "weights" , nextLine(finp) , 2 , 2 ); // Wczytuje domyślny przedział wag, będzie użyte gdy nie podano innej wartości dla wagi.
    gmin_w = tmp_f[0];
    gmax_w = tmp_f[1];
    PEXCP( gmin_w <= gmax_w );

    weights.resize(size_weights);      // Pamięć na wagi.
    min_w.fill(gmin_w,size_weights);    // Pamięć na minimalne ograniczenia wag, inicjujemy min_w.
    max_w.fill(gmax_w,size_weights);    // Pamięć na maksymalne ograniczenia wag, inicjujemy max_w.

    for( nnityp i=0 ; i<size_weights ; i++ ) {     // Po wszystkich wagach.
        weights[i] = rnd.getF(gmin_w,gmax_w);    // Losujemy wagi z domyślnego przedziału.
    }

    QString line;

    while( (line = nextLine(finp)) != "endweights" ) { // Wczytuj definicje wag aż do napotkania endWeight.
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

        // wejścia neuronu.
        neuron.idx_i = getInts("inp",finp,1);
        for( int i=0 ; i<neuron.idx_i.size() ; i++ ) {
            PEXCP( neuron.idx_i[i] >= 0 );
            PEXCP( neuron.idx_i[i] < nr + size_i );
        }

        // zakres indeksów wejść neuronu
        tmp_i = getInts("rng_inputs",finp,2,2);
        neuron.min_i = tmp_i[0];
        neuron.max_i = tmp_i[1];
        PEXCP( neuron.max_i - neuron.min_i + 1 >= neuron.idx_i.size() ); // wejścia muszą być różne, więc przedział musi być większy lub równy niż ilość wejść

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

    fout << "size_inputs:  "  << size_i         << QString::fromUtf8(" # rozmiar wejścia") << endl;
    fout << "size_outputs: "  << size_o         << QString::fromUtf8(" # rozmiar wyjścia") << endl;
    fout << "size_neurons: "  << neurons.size() << QString::fromUtf8(" # ilość neuronów")  << endl;
    fout << "size_weights: "  << weights.size() << QString::fromUtf8(" # ilość wag")       << endl;
    fout << "size_convols: "  << convs.size()   << QString::fromUtf8(" # ilość splotów")   << endl;
    fout << endl;


    fout << "weights_penal: "  << forcesign << weights_penal << QString::fromUtf8(" # kara za duże wagi")    << endl;
    fout << "signal_penal:  "  << forcesign << signal_penal  << QString::fromUtf8(" # kara za duże sygnały") << endl;
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

// Zaburza wartość losowej wagi.
bool NNNet::chaosWeight( FRnd &rnd, cftyp strength ) {
    TVInt v;                                      // Indeksy wag które można zaburzyć.
    for( nnityp i=0 ; i<weights.size() ; i++ ) {  // Po wszystkich wagach.
        if( max_w[i] - min_w[i] > 0.000001 )      // Jeśli jest jakiś sensowny przedział, to
            v.append(i);                          // wagę można modyfikować.
    }
    if( v.size() == 0 )                           // Jeśli żadna waga nie nadaje się do modyfikacji, to
        return false;                             // Nie udało się zaburzyć żadnej wagi.
    nncityp r = v[ rnd() % v.size() ];            // Wylosuj wagę.
    rnd.chaos( weights[r] , min_w[r] , max_w[r] , strength ); // Zaburz wagę.
    return true;                                  // Udało się zaburzyć wagę.
}

// Zaburza wartość wszystkich wag
void NNNet::chaosWeights( FRnd &rnd, cftyp strength ) {
    for( nnityp i=0 ; i<weights.size() ; i++ ) {                      // Po wszystkich wagach.
        if( max_w[i] - min_w[i] > 1E-6 ) {                            // Jeśli jest jakiś sensowny przedział, to
            rnd.chaos( weights[i] , min_w[i] , max_w[i] , strength ); // Zaburz wagę.
        }
    }
}


// Zaburza losowo indeks wagi w losowo wybranym neuronie.
bool NNNet::chaosIdxWeight( FRnd &rnd ) {
    NNNeuron &n = neurons[ rnd() % neurons.size() ];  // Wylosuj neuron.
    nncityp idxw = rnd.getI( n.min_w , n.max_w );     // Wylosuj indeks wagi z dozwolonego przedziału wag.
    nncityp pos = rnd() % n.idx_w.size();             // Wylosuj pozycję na którą wstawić nowy indeks.
    n.idx_w[pos] = idxw;                              // Wstaw nowy indeks.
    if( n.isConvolution() ) {                         // Jeśli neuron jest w splocie z innymi neuronami.
        CTVInt &conv = convs[n.conv];                 // Wektor neuronów splecionych.
        for( nnityp i=0 ; i<conv.size() ; i++ ) {     // We wszystkich neuronów ze splotu
            neurons[ conv[i] ].idx_w[pos] = idxw;     // na identycznej pozycji zmień identyczny indeks.
        }
    }
    return true;                                      // Zwróć informację że udalo się zaburzyć indeks wagi w neuronie.
}

// Zaburza losowo indeks wejścia neuronu. Zakładamy że
// wejścia przed zaburzeniem i po zaburzeniu są unikalne,
// nie powtarzają się.
bool NNNet::chaosIdxInput( FRnd &rnd ) {
    TVInt v;                                           // Miejsce na różne indeksy.
    for( nnityp i=0 ; i<neurons.size() ; i++ ) {       // Po wszystkich neuronach.
        CNNNeuron &n = neurons[i];                     // Tylko skrót do neuronu.
        if( n.idx_i.size() < n.max_i - n.min_i + 1 )   // Jeśli ilość wejść jest mniejsza niż maksymalna ilość wejść, to
            v.append(i);                               // jest możliwość zaburzenia z utrzymaniem unikalności.
    }
    if( v.size() == 0 )                                // Jeśli żadnego neuronu nie można zaburzyć, to
        return false;                                  // zwróć informację, że nie udało się zaburzenie.
    NNNeuron &n = neurons[ v[ rnd() % v.size() ] ];    // Wylosuj neuron z neuronów nadających się do modyfikacji.
    QSet<nnityp> idx_i;                                // Zbiór na istniejące wejścia neuronu.
    for( nnityp i=0 ; i<n.idx_i.size() ; i++ ) {       // Dodaj do zbioru
        idx_i.insert( n.idx_i[i] );                    // wszystkie istniejące wejścia.
    }
    v.clear();                                         // Teraz v przyda się na wejścia których neuron nie ma.
    for( nnityp i=n.min_i ; i<=n.max_i ; i++ ) {       // Iteruj po wejściach jakie neuron może mieć.
        if( ! idx_i.contains(i) )                      // Jeśli neuron nie ma wejścia, to
            v.append(i);                               // dodaj to wejście.
    }
    nncityp inp = v[ rnd() % v.size() ];               // Wylosuj nowe wejście dla neuronu.
    nncityp pos = rnd() % n.idx_i.size();              // Wylosuj pozycję na której stare wejście zostanie usunięte.
    EXCP( n.idx_i[pos] != inp );                       // Nie dublujemy wejść, uproszczona asercja.
    n.idx_i[pos] = inp;                                // Zastąp stare wejście nowym, losowym.
    return true;                                       // Udało się zaburzenie.
}

// Zaburzenie losowe w sieci neuronowej.
bool NNNet::chaos(
    FRnd &rnd ,     // generator liczb losowych
    cftyp strength ,  // siła zaburzenia w przypadku zaburzania wartości (nie indeksu) wagi.
    const bool weights,
    const bool idx_weights,
    const bool idx_input
) {
    for( nnityp i=0 ; i<1000 ; i++ ) {                 // Próbuj 1000 razy wykonać jakieś zaburzenie losowe.
        TVInt v{0,1,2};                                // Są trzy możliwosci zaburzeń losowych.
        while( v.size() > 0 ) {                        // Do póki nie wykorzystano wszystkich możliwości.
            nncityp r = rnd() % v.size();              // Wylosuj możliwość, a właściwie to indeks możliwości.
            switch( v[ r ] ) {                         // Skok do procedury zaburzenia
                case 0:
                    if( weights && chaosWeight(rnd,strength) )    // Jak się udało zaburzenie, to zwroć informację, że się udało.
                        return true;
                    break;
                case 1:
                    if( idx_weights &&  chaosIdxWeight(rnd) )           // Jak wyżej.
                        return true;
                    break;
                case 2:
                    if( idx_input && chaosIdxInput(rnd) )             // Jak wyżej.
                        return true;
                    break;
            }
            v.remove(r);                                 // Usuń możliwość spod indeksu r i spróbuj następnej możliwości.
        }
    }
    return false;                                        // Nie udało się zaburzyć pomimo 1000 prób.
}

}
