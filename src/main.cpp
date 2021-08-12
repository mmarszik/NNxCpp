#include <QVector>
#include <cmath>
#include <QTextStream>
#include <QFile>
#include <climits>

#include "src/engine/nnnet.h"
#include "src/engine/data.h"
#include "src/misc/rand.h"

using namespace NsNet;

double classify(CNNData &data, const NNNet &nn) {

    int rightClass = 0;

    for( int i=0 ; i<data.size() ; i++ ) {
        CTVFlt obuf = nn.compute( data[i].getInps() );

        int max1 = 0;
        for( int j=1 ; j<obuf.size() ; j++ ) {
            if( obuf[j] > obuf[max1] ) {
                max1 = j;
            }
        }

        int max2 = 0;
        for( int j=1 ; j<data[i].sizeOut() ; j++ ) {
            if( data[i].getOut(j) > data[i].getOut(max2) ) {
                max2 = j;
            }
        }
        rightClass += max1 == max2;

    }

    const double ratio = 100.0 * rightClass / data.size();
//    {
//        QTextStream stdo(stdout);
//        stdo << "---------------------------------------" << endl;
//        stdo << "rightClass:" << ratio << "%" << endl;
//        stdo << "---------------------------------------" << endl;
//    }

    return ratio;
}

struct NNValue {
    NNNet nn;
    double value;
    double classify;
    bool force;
};



void fromStart() {
    QTextStream stdOut(stdout);

    NNData learn, test;
    {
        FRnd rnd(1);
        CNNData data = NNData::mkData(false,0,4,0,3,5,"/home/m/tmp/test_data.csv",",");
        data.split( learn , 80000 , test, data.size()-80000, rnd );
    }

    FRnd rnd(1);

    const time_t start = time(NULL);
    nnftyp bigPenal = 1E-12;

    NNNet nn;
    nn.read("nndef.txt" );
    stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.toUniqueWeights(1);
    stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "nndef_out.txt" );


    {
        CNNData subLearn = learn.rndSelect(5000,rnd);
        nnftyp e = nn.error( subLearn, bigPenal);
        for( nnityp loop=1 ; loop <= 10000 ; loop ++ ) {
            NNNet tmpNN;
            tmpNN.read("nndef.txt" );
//            tmpNN.randWeights(rnd,-1.0,+1.0);
            tmpNN.randIdxW(rnd);
            nncftyp tmpE = tmpNN.error( subLearn , bigPenal );
            if( tmpE < e || loop==1 ) {
                e = tmpE;
                nn = tmpNN;
                stdOut << "init loop=" << loop << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
            }
        }
        stdOut << "learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );
    }


    {
        TVFlt toLearn( nn.sizeWeights() , 1.0 );
        toLearn[0] = 0;

        nnityp learnSize = 1000;
        nnityp learnTime = 50;

        for( nnityp loop=1 ; true ; loop++ ) {
            {
                CNNData subLearn = learn.rndSelect(learnSize,rnd);
                nn.learnRand1( subLearn, bigPenal, learnTime, 0, 1E-12, 0.01 , loop==1 ? 1 : 0.1 , rnd(), true, false, false );
                stdOut << "learnRand1; loop=" << loop;
                stdOut << " time=" << learnTime;
                stdOut << " size=" << learnSize;
                stdOut << " learn error=" << nn.error( learn , bigPenal );
                stdOut << " test=" << classify( test ,nn ) << "%";
                stdOut << " time=" << (time(NULL)-start) << "s" << endl;
                nn.save("nndef_out.txt" );
            }

            {
                CNNData subLearn = learn.rndSelect(learnSize,rnd);
                nn.momentum2( subLearn , CNNData(), learnTime, 0, 0.01, 0.1, 1E-12, 0.95, toLearn, bigPenal, 5, nullptr);
                stdOut << "momentum2;  loop=" << loop;
                stdOut << " time=" << learnTime;
                stdOut << " size=" << learnSize;
                stdOut << " learn error=" << nn.error( learn , bigPenal );
                stdOut << " test=" << classify( test ,nn ) << "%";
                stdOut << " time=" << (time(NULL)-start) << "s" << endl;
                nn.save( "nndef_out.txt" );
            }

            {
                if( learnSize < 20000 ) {
                    learnSize += 1000;
                } else {
                    learnSize += 5000;
                }
                if( learnSize > learn.size() ) {
                    learnSize = learn.size();
                }
            }

            {
                learnTime += 50;
                if( learnTime > 1000 ) {
                    learnTime = 1000;
                }
            }

        }
    }


}

void nextLearn() {
    QTextStream stdOut(stdout);

    NNData learn, test;
    {
        FRnd rnd(1);
        CNNData data = NNData::mkData(false,0,4,0,3,5,"/home/m/tmp/test_data.csv",",");
        data.split( learn , 80000 , test, data.size()-80000, rnd );
    }

    FRnd rnd(6);

    const time_t start = time(NULL);
    nnftyp bigPenal = 1E-12;

    NNNet nn;
    nn.read("nndef.txt" );
    stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;

    nn.toUniqueWeights( 1 );
    stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "nndef_out.txt" );


    {
        TVFlt toLearn( nn.sizeWeights() , 1.0 );
        toLearn[0] = 0;

        for( nnityp loop=1 ; true ; loop++ ) {
            {
                nn.learnRand1( learn, bigPenal, 1200 , 0, 1E-12, 0.05 , 0.30 , rnd(), true, false, false );
                stdOut << "learnRand1; loop=" << loop;
                stdOut << " learn error=" << nn.error( learn , bigPenal );
                stdOut << " test=" << classify( test ,nn ) << "%";
                stdOut << " time=" << (time(NULL)-start) << "s" << endl;
                nn.save("nndef_out.txt" );
            }

            if( loop >= 3 )
            {
                nn.momentum2( learn , CNNData(), 1200, 0, 0.01, 0.1, 1E-12, 0.95, toLearn, bigPenal, 5, nullptr);
                stdOut << "momentum2;  loop=" << loop;
                stdOut << " learn error=" << nn.error( learn , bigPenal );
                stdOut << " test=" << classify( test ,nn ) << "%";
                stdOut << " time=" << (time(NULL)-start) << "s" << endl;
                nn.save( "nndef_out.txt" );
            }

        }
    }


}


void experiment0() {
    QTextStream stdOut(stdout);

    NNData learn, test;
    {
        FRnd rnd(1);
        CNNData data = NNData::mkData(false,0,4,0,3,5,"/home/m/tmp/test_data.csv",",");
        data.split( learn , 80000 , test, data.size()-80000, rnd );
    }

    FRnd rnd(6);

    const time_t start = time(NULL);
    nnftyp bigPenal = 1E-12;

    NNNet nn;
    nn.read("nndef.txt" );
    stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.toUniqueWeights(1);
    stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "nndef_out.txt" );

    {
        NNNet orgNN;
        orgNN.read("nndef.txt" );
        orgNN.toUniqueWeights(1);
        orgNN.setMinWeights(-30);
        orgNN.setMaxWeights(+30);
        CNNData subLearn = learn.rndSelect(learn.size()/16,rnd);
        nnftyp e = nn.error( subLearn, bigPenal);
        for( nnityp loop=1 ; loop <= 30000 ; loop ++ ) {
            NNNet tmpNN = orgNN;
            tmpNN.randWeights(rnd,-0.333,+0.333);
            nncftyp tmpE = tmpNN.error( subLearn , bigPenal );
            if( tmpE < e || loop==1 ) {
                e = tmpE;
                nn = tmpNN;
                stdOut << "init loop=" << loop << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
                nn.save( "nndef_out.txt" );
            }
        }
        stdOut << "learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    }

    for( nnityp i=16 ; i>=1 ; i-- ) {
        nn.annealing( learn.rndSelect(learn.size()/i,rnd) , bigPenal , 3600*5/16 , 0.00001 , 0.0500 , 0.5, rnd() );
        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );
    }

//    for( nnityp i=4 ; i!=0 ; i/=2 ) {
//        nn.learnRand1( learn.rndSelect(learn.size()/i,rnd) , bigPenal , 3600 , 0 , 1E-6 , 1E-6, 1 , 1 , true , false , true );
//        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
//        nn.save( "nndef_out.txt" );
//    }

    {
        TVFlt toLearn( nn.sizeWeights() , 1.0 );
        toLearn[0] = 0;

        for( nnityp loop=1 ; true ; loop++ ) {

            if( loop >= 1 )
            {
                nn.momentum2( learn , CNNData(), 1200, 0, 0.01, 0.1, 1E-12, 0.99, toLearn, bigPenal, 5, nullptr);
                stdOut << "momentum2;  loop=" << loop;
                stdOut << " learn error=" << nn.error( learn , bigPenal );
                stdOut << " test=" << classify( test ,nn ) << "%";
                stdOut << " time=" << (time(NULL)-start) << "s" << endl;
                nn.save( "nndef_out.txt" );
            }

            {
                nn.learnRand1( learn, bigPenal, 1200 , 0, 1E-12, 0.025 , 0.300 , rnd(), true, false, false );
                stdOut << "learnRand1; loop=" << loop;
                stdOut << " learn error=" << nn.error( learn , bigPenal );
                stdOut << " test=" << classify( test ,nn ) << "%";
                stdOut << " time=" << (time(NULL)-start) << "s" << endl;
                nn.save("nndef_out.txt" );
            }

        }
    }


}


int main(int argc, char *argv[])
{
    experiment0();
    return 0;
}


