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


// 3-5-5 - learnRand1; loop=4 learn error=0.318215 test=64.275% time=20349s
// 3-6-5 - momentum2;  loop=3 learn error=0.294996 test=67.595% time=16767s
// 3-7-5 - momentum2;  loop=8 learn error=0.266593 test=73.475% time=31857s
void fromStart() {
    QTextStream stdOut(stdout);

    NNData learn, test;
    {
        FRnd rnd(1);
        CNNData data = NNData::mkData(false,0,4,0,3,5,"/home/m/tmp/test_data.csv",",");
        data.split( learn , 80000 , test, data.size()-80000, rnd() );
    }

    const unsigned long long rndSeed = std::random_device()();
    stdOut << "rndSeed=" << rndSeed << endl;
    FRnd rnd(rndSeed);

    const time_t start = time(NULL);
    nnftyp bigPenal = 1E-8;

    NNNet nn;
    nn.read("nndef.txt" );
    stdOut << "learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.toUniqueWeights(0);
    stdOut << "learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.randWeights(rnd,-1.0,+1.0);
    nn.save( "nndef_out.txt" );

    if( true )
    {
        NNNet orgNN;
        orgNN.read("nndef.txt" );
        orgNN.toUniqueWeights(1);
        orgNN.setMinWeights(-30);
        orgNN.setMaxWeights(+30);
        CNNData subLearn = learn.rndSelect(500,rnd());
        nnftyp e = nn.error( subLearn, bigPenal);
        for( nnityp loop=1 ; loop <= 50000 ; loop ++ ) {
            NNNet tmpNN = orgNN;
            tmpNN.randWeights(rnd,-0.1,+0.1);
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

    TVFlt toLearn( nn.sizeWeights() , 1.0 );
//    toLearn[0] = 0;


    {
        TVInt parts = { 500,1000,1500,2000,3500,5000,10000,20000,30000,learn.size()/2};
        for( nnityp loop=0 ; loop<parts.size() ; loop++ ) {
            NNData subLearn,subTest;
            PEXCP( parts[loop]*2 <= learn.size() );
            if( parts[loop]*4 <= learn.size() ) {
                learn.split(subLearn,parts[loop],subTest,parts[loop]*3,rnd());
            } else if( parts[loop]*3 <= learn.size() ) {
                learn.split(subLearn,parts[loop],subTest,parts[loop]*2,rnd());
            } else {
                learn.split(subLearn,parts[loop],subTest,parts[loop]*1,rnd());
            }
            NNNet best = nn;
            nncftyp maxStep   =  parts[loop] < 500 ? 1E-2 : parts[loop] <= 1000 ? 5E-2 : 1E-1;
            nncftyp bigPenal2 =  bigPenal;

            nn.momentum2( subLearn , subTest, 3600*24, 30, 1E-6, maxStep, 1E-9, 0.80, toLearn, bigPenal2, 5, &best);
            nn = best;
            stdOut << "momentum2;  loop=" << loop;
            stdOut << " part=" << parts[loop];
            stdOut << " learn error=" << nn.error( learn , bigPenal );
            stdOut << " test=" << classify( test ,nn ) << "%";
            stdOut << " time=" << (time(NULL)-start) << "s" << endl;
            nn.save( "nndef_out.txt" );
        }
        bigPenal /= 5;
        if( bigPenal < 1E-12 ) {
            bigPenal = 1E-12;
        }
    }


    {
        for( nnityp loop=1 ; true ; loop++ ) {
            {
                nn.learnRand1( learn, bigPenal, 600, 0, 1E-12, 1E-6 , 1E-1 , rnd(), true, false, false );
                stdOut << "learnRand1; loop=" << loop;
                stdOut << " learn error=" << nn.error( learn , bigPenal );
                stdOut << " test=" << classify( test ,nn ) << "%";
                stdOut << " time=" << (time(NULL)-start) << "s" << endl;
                nn.save("nndef_out.txt" );
            }

            {
                nn.momentum2( learn , CNNData(), 3000, 0, 1E-3, 1E-3, 1E-9, 0.5, toLearn, bigPenal, 5, nullptr);
                stdOut << "momentum2;  loop=" << loop;
                stdOut << " learn error=" << nn.error( learn , bigPenal );
                stdOut << " test=" << classify( test ,nn ) << "%";
                stdOut << " time=" << (time(NULL)-start) << "s" << endl;
                nn.save( "nndef_out.txt" );
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
        data.split( learn , 80000 , test, data.size()-80000, rnd() );
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

            if( loop >= 1 )
            {
                nn.momentum2( learn , CNNData(), 3000, 0, 5E-2, 5E-2, 1E-12, 0.995, toLearn, bigPenal, 5, nullptr);
                stdOut << "momentum2;  loop=" << loop;
                stdOut << " learn error=" << nn.error( learn , bigPenal );
                stdOut << " test=" << classify( test ,nn ) << "%";
                stdOut << " time=" << (time(NULL)-start) << "s" << endl;
                nn.save( "nndef_out.txt" );
            }

            if( loop >= 1 )
            {
                //nn.annealing(learn,bigPenal,1800,1E-6,1E-3,0.4,rnd());
                nn.learnRand1(learn,bigPenal,600,0,1E-9,1E-6,1E-2,rnd(),true,false,false);
                stdOut << "learnRand1; loop=" << loop;
                stdOut << " learn error=" << nn.error( learn , bigPenal );
                stdOut << " test=" << classify( test ,nn ) << "%";
                stdOut << " time=" << (time(NULL)-start) << "s" << endl;
                nn.save("nndef_out.txt" );
            }

        }
    }


}


void experiment0(const bool reLearn) {
    QTextStream stdOut(stdout);

    NNData learn, test;
    {
        FRnd rnd(1);
        CNNData data = NNData::mkData(false,0,4,0,3,5,"/home/m/tmp/test_data.csv",",");
        data.split( learn , 80000 , test, data.size()-80000, rnd() );
    }

    const unsigned long long rndSeed = std::random_device()();
    stdOut << "rndSeed=" << rndSeed << endl;
    FRnd rnd(rndSeed);

    const time_t start = time(NULL);
    nnftyp bigPenal = 1E-9;

    NNNet nn;
    nn.read("nndef.txt" );
    stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;

//    nn.toUniqueWeights(0);
//    stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
//    nn.save( "nndef_out.txt" );

//    nn.randWeights(rnd,-0.1,+0.1);
    nn.randIdxW( rnd );
    stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "nndef_out.txt" );


    nn.forceIdx(learn,bigPenal,1E4,true);
    stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "nndef_out.txt" );


    NNNet best = nn;
    nnftyp bestError = best.error( learn, bigPenal);

    for( nnityp loop=1 ; true ; loop++ ) {

        nncftyp oldError = nn.error( learn, bigPenal );

        {
            nn.learnRand1( learn, bigPenal,  1800, 0, 1E-12, 0.001 , 0.005 , rnd(), true, true, false );
            stdOut << "learnRand1; loop=" << loop;
            stdOut << " learn_error=" << nn.error( learn , bigPenal );
            stdOut << " best_error=" << best.error( learn , bigPenal );
            stdOut << " test=" << classify( test ,nn ) << "%";
            stdOut << " best_test=" << classify( test ,best ) << "%";
            stdOut << " time=" << (time(NULL)-start) << "s" << endl;
        }

        {
            nn.momentum2( learn , CNNData(), 1800, 0, 0.001, 0.1, 1E-12, 0.99, CTVFlt(), bigPenal, 1, nullptr);
            stdOut << "momentum2;  loop=" << loop;
            stdOut << " learn_error=" << nn.error( learn , bigPenal );
            stdOut << " best_error=" << best.error( learn , bigPenal );
            stdOut << " test=" << classify( test ,nn ) << "%";
            stdOut << " best_test=" << classify( test ,best ) << "%";
            stdOut << " time=" << (time(NULL)-start) << "s" << endl;
        }

        if( loop == 1 ) {
            nn.toUniqueWeights();
        }

        nncftyp newError = nn.error( learn, bigPenal );

        if( newError < bestError ) {
            best = nn;
            bestError = newError;
            best.save( "nndef_out.txt" );
        }

        if( oldError - newError < 5E-5 ) {
            nn = best;
            nncftyp strength = rnd.getF( 0.01  ,  0.5 );
            nncityp loops    = rnd.getI(    4  ,   20 );
            nncityp rndSeed  = rnd();
            nn.chaosWeights(rndSeed, strength, loops );
            stdOut << "chaos; loop=" << loop;
            stdOut << " strength=" << strength;
            stdOut << " loops=" << loops;
            stdOut << " rndSeed=" << rndSeed;
            stdOut << " learn_error=" << nn.error( learn , bigPenal );
            stdOut << " best_error=" << best.error( learn , bigPenal );
            stdOut << " test=" << classify( test ,nn ) << "%";
            stdOut << " best_test=" << classify( test ,best ) << "%";
            stdOut << " time=" << (time(NULL)-start) << "s" << endl;
        }

    }

}




// 3-8-5 - momentum2;  loop=66 learn error=0.25352 test=74.985% time=59613s
// 3-8-5 - momentum2;  parts=70000 loop=30 learn error=0.258517 test=73.855% time=56426s
// 3-10-5 - momentum2;  parts=60000 loop=1 bigPenal=1e-09 maxStep=0.2 maxFails=200 learn error=0.242603 test=75.925% time=967s
// 3-10-10-5 - momentum2;  parts=60000 loop=2 bigPenal=1e-09 maxStep=0.2 maxFails=200 learn error=0.105752 test=89.985% time=2713s
// 3-16-16-5 - momentum2;  parts=60000 loop=1 bigPenal=1e-09 maxStep=0.2 maxFails=200 learn error=0.0550948 test=95.625% time=4699s
// 3-16-16-5 - momentum2;  parts=60000 loop=3 bigPenal=1e-06 maxStep=0.05 maxFails=500 learn error=0.086473 test=92.9% time=8808s
// 3-16-16-5 - momentum2;  parts=60000 loop=1 bigPenal=1e-08 maxStep=0.05 maxFails=500 learn error=0.0714181 test=93.64% time=17583s
// 3-16-16-5 - momentum2;  parts=60000 loop=1 bigPenal=1e-09 maxStep=0.05 maxFails=500 learn error=0.0805478 test=92.73% time=15426s
// 3-16-16-5 - momentum2;  parts=80000 loop=1 bigPenal=1e-09 maxStep=0.05 maxFails=500 learn error=0.0604753 test=94.955% time=19065s

void experiment1( const bool reLearn ) {
    QTextStream stdOut(stdout);

    NNData learn, test;
    {
        FRnd rnd(1);
        CNNData data = NNData::mkData(false,0,4,0,3,5,"/home/m/tmp/test_data.csv",",");
        data.split( learn , 80000 , test, data.size()-80000, rnd() );
    }

    const unsigned long long rndSeed1 = std::random_device()();
    const unsigned long long rndSeed2 = std::random_device()();
    stdOut << "rndSeed1=" << rndSeed1 << endl;
    stdOut << "rndSeed2=" << rndSeed2 << endl;
    FRnd rnd(rndSeed1);

    const time_t start = time(NULL);
    nnftyp bigPenal = 1E-7;

    NNNet nn;
    nn.read("nndef.txt" );
    stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;

    if( ! reLearn )
    {
        nn.toUniqueWeights(0);
        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );
    }

    if( ! reLearn )
    {
        nn.randWeights(rnd,-0.1,+0.1);
        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );
    }

    {
        nnftyp maxFails = 500;
        nnftyp maxStep = 0.001;
        CTVInt parts = { 1000, 2000, 3000, 5000, 7500, 10000, 20000, 40000, learn.size() };
        for( nnityp p=0 ; p<parts.size() ; p++ ) {
            NNData subLearn, subTest;
            learn.split( subLearn, parts[p] , subTest , parts[p]*2 , rndSeed2+p );
            for( nnityp loop=1 ; true ; loop++ ) {

                nncftyp er1 = nn.error(subLearn,bigPenal);

                NNNet bestNN = nn;
                nn.momentum2(subLearn , subTest, 1800, maxFails, maxStep/100, maxStep, 1E-12, 0.97, CTVFlt(), bigPenal, 1, &bestNN);
                nn = bestNN;

                stdOut << "momentum2;  parts=" << parts[p];
                stdOut << " loop=" << loop;
                stdOut << " bigPenal=" << bigPenal;
                stdOut << " maxStep=" << maxStep;
                stdOut << " maxFails=" << maxFails;
                stdOut << " learn error=" << nn.error( learn , bigPenal );
                stdOut << " test=" << classify( test ,nn ) << "%";
                stdOut << " time=" << (time(NULL)-start) << "s" << endl;
                nn.save( "nndef_out.txt" );

                nncftyp er2 = nn.error(subLearn,bigPenal);
                if( er1 - er2 < 1E-6 ) {
                    break;
                }

            }
            bigPenal *= 0.1;
            maxStep  *= 1.77827941003892;
            maxFails *= 1.33352143216332;
        }
    }

}


// 3-8-5 - momentum2;  loop= 5 learn error=0.258124 test=74.895% time=6022s
// 3-8-5 - momentum2;  loop=28 learn error=0.253949 test=75.06% time=33702s
// 3-8-5 - momentum2;  loop=11 learn error=0.254929 test=74% time=13241s
// 3-9-5 - momentum2;  loop=5 learn error=0.248381 test=75.8% time=6102s
// 3-9-5 - momentum2;  loop=7 learn error=0.248409 test=74.895% time=12611s
void experiment2( const bool reLearn ) {
    QTextStream stdOut(stdout);

    NNData learn, test;
    {
        FRnd rnd(1);
        CNNData data = NNData::mkData(false,0,4,0,3,5,"/home/m/tmp/test_data.csv",",");
        data.split( learn , 80000 , test, data.size()-80000, rnd() );
    }

    const unsigned long long rndSeed1 = std::random_device()();
    const unsigned long long rndSeed2 = std::random_device()();
    stdOut << "rndSeed1=" << rndSeed1 << endl;
    stdOut << "rndSeed2=" << rndSeed2 << endl;
    FRnd rnd(rndSeed1);

    const time_t start = time(NULL);
    nnftyp bigPenal = 1E-6;

    NNNet nn;
    nn.read("nndef.txt" );
    stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;

    if( ! reLearn )
    {
        nn.toUniqueWeights(1);
        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );
    }

    if( ! reLearn )
    {
        nn.randWeights(rnd,-0.4,+0.4);
        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );
    }

    if( false )
    {
        nn.learnRand1( learn.rndSelect(8000,rndSeed2), bigPenal, 3600 , 0, 0, 0.001, 0.3 , rnd(), true, false, false );
        stdOut << "learnRand1;  loop=" << 0;
        stdOut << " learn error=" << nn.error( learn , bigPenal );
        stdOut << " test=" << classify( test ,nn ) << "%";
        stdOut << " time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );
    }

    for( nnityp loop=1 ; true ; loop++ ) {
        nncftyp er1 = nn.error(learn,bigPenal);

        nn.momentum2(learn , CNNData(), 1200, 0, 1E-2, 0.05, 1E-14, 0.9, CTVFlt(), bigPenal, 1, NULL);
        stdOut << " momentum2;  loop=" << loop;
        stdOut << " learn error=" << nn.error( learn , bigPenal );
        stdOut << " test=" << classify( test ,nn ) << "%";
        stdOut << " time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );

        nn.learnRand1( learn, bigPenal, 1200 , 0, 0, 0.001, 0.1 , rnd(), true, false, false );
        stdOut << "learnRand1;  loop=" << loop;
        stdOut << " learn error=" << nn.error( learn , bigPenal );
        stdOut << " test=" << classify( test ,nn ) << "%";
        stdOut << " time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );


        nncftyp er2 = nn.error(learn,bigPenal);
        if( er1 - er2 < 1E-5 ) {
            break;
        }

        bigPenal *= 0.1;
        if( bigPenal < 1E-14 ) {
            bigPenal = 1E-14;
        }

    }

}


void experiment3( const bool reLearn ) {
    QTextStream stdOut(stdout);

    NNData learn, test;
    {
        FRnd rnd(1);
        CNNData data = NNData::mkData(false,0,4,0,3,5,"/home/m/tmp/test_data.csv",",");
        data.split( learn , 80000 , test, data.size()-80000, rnd() );
    }

    const unsigned long long rndSeed1 = std::random_device()();
    const unsigned long long rndSeed2 = std::random_device()();
    stdOut << "rndSeed1=" << rndSeed1 << endl;
    stdOut << "rndSeed2=" << rndSeed2 << endl;
    FRnd rnd(rndSeed1);

    const time_t start = time(NULL);
    nnftyp bigPenal = 1E-12;

    NNNet nn;
    nn.read("nndef.txt" );
    stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;

    if( ! reLearn )
    {
        nn.randIdxW(rnd,-1,+1);
        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );
    }

    if( ! reLearn )
    {
        nn.forceIdx(learn,bigPenal,1E4,true);
        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );
    }

//    if( ! reLearn )
//    {
//        nn.forcePairIdx(learn,bigPenal,1E4,true,1);
//        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
//        nn.save( "nndef_out.txt" );
//    }

    if( ! reLearn )
    {
        nn.toUniqueWeights(0);
        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );
    }

//    if( ! reLearn )
//    {
//        nn.randWeights(rnd,-0.01,+0.01);
//        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
//        nn.save( "nndef_out.txt" );
//    }

    for( nnityp loop=1 ; true ; loop++ ) {
        nncftyp er1 = nn.error(learn,bigPenal);
        nn.momentum2(learn , CNNData(), 1800, 0, 1E-1, 1, 1E-14, 0.9, CTVFlt(), bigPenal, 1, NULL);
        nncftyp er2 = nn.error(learn,bigPenal);
        stdOut << "momentum2;  loop=" << loop;
        stdOut << " learn error=" << nn.error( learn , bigPenal );
        stdOut << " test=" << classify( test ,nn ) << "%";
        stdOut << " time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );
        if( er1 - er2 < 1E-5 ) {
            break;
        }
    }

}

//
//3-7
//subLearn.size=80000 nn.sizeWeights=39 learn error=0.321294 class test=63.765% class learn=63.8537% time=14818s
void experimentA() {
    QTextStream stdOut(stdout);

    const int learnSize = 80000;
    NNData learn, test;
    {
        FRnd rnd(1);
        CNNData data = NNData::mkData(false,0,4,0,3,5,"/home/m/tmp/test_data.csv",",");
        data.split( learn , learnSize, test, data.size()-learnSize, rnd() );
    }

//  const unsigned long long rndSeed = 2634215374;
    const unsigned long long rndSeed = std::random_device()();

    stdOut << "rndSeed=" << rndSeed << endl;
    FRnd rnd( rndSeed );

    NNData subLearn = learn; //learn.rndSelect( 10000 , 2 );

    const time_t start = time(NULL);
    nnftyp bigPenal = 1E-8;

    NNNet nn;
    nn.read("nndef.txt" );
    stdOut << "subLearn.size=" << subLearn.size() << " nn.sizeWeights=" << nn.sizeWeights() << " learn error=" << nn.error( subLearn , bigPenal ) << " class test=" << classify( test ,nn ) << "% class learn=" << classify( subLearn ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;

//    nn.randWeights( rnd , -1.0 , +1.0 );
//    nn.randInputs( rnd );
    nn.randIdxW( rnd );
    stdOut << "subLearn.size=" << subLearn.size() << " nn.sizeWeights=" << nn.sizeWeights() << " learn error=" << nn.error( subLearn , bigPenal ) << " class test=" << classify( test ,nn ) << "% class learn=" << classify( subLearn ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "nndef_out.txt" );

    if( true )
    {
        nnftyp error = nn.error( subLearn , bigPenal );
        for( int loop=1 ; loop<=10000 ; loop++ ) {
            NNNet NNtmp = nn;
//            NNtmp.randWeights( rnd , -1.0 , +1.0 );
//            NNtmp.randInputs(rnd);
            NNtmp.randIdxW( rnd );
            nncftyp tmp = NNtmp.error( subLearn , bigPenal );
            if( tmp < error ) {
                nn = NNtmp;
                error = tmp;
                stdOut << loop << "] subLearn.size=" << subLearn.size() << " nn.sizeWeights=" << nn.sizeWeights() << " learn error=" << nn.error( subLearn , bigPenal ) << " class test=" << classify( test ,nn ) << "% class learn=" << classify( subLearn ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
            }
        }
        nn.save( "nndef_out.txt" );
    }

    nn.forceIdx(subLearn,bigPenal,1E4,true,1);
    stdOut << "subLearn.size=" << subLearn.size() << " nn.sizeWeights=" << nn.sizeWeights() << " learn error=" << nn.error( subLearn , bigPenal ) << " class test=" << classify( test ,nn ) << "% class learn=" << classify( subLearn ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "nndef_out.txt" );

    for( int i=0 ; i<200 ; i++ ) {
        NNNet nnTmp = nn;
        {
            int change = 0;
            do {
                change += nnTmp.chaos( rnd , 1 , false , true , false );
            } while( change == 0 || rnd() % 2 );
        }
        nnTmp.forceIdx(subLearn, bigPenal, 20, true, 1);
        if( nnTmp.error(subLearn) <= nn.error(subLearn) ) {
            nn = nnTmp;
            nn.save( "nndef_out.txt" );
            stdOut << "Increase!!!" << endl;
        }
        stdOut << "loop=" << i << "] subLearn.size=" << subLearn.size() << " nn.sizeWeights=" << nnTmp.sizeWeights() << " learn error=" << nnTmp.error( subLearn , bigPenal ) << " class test=" << classify( test ,nnTmp ) << "% class learn=" << classify( subLearn ,nnTmp ) << "% time=" << (time(NULL)-start) << "s" << endl;
    }


    nn.doubleRndIdxWeightForce(subLearn,3600,0,20,1);
    stdOut << "subLearn.size=" << subLearn.size() << " nn.sizeWeights=" << nn.sizeWeights() << " learn error=" << nn.error( subLearn , bigPenal ) << " class test=" << classify( test ,nn ) << "% class learn=" << classify( subLearn ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "nndef_out.txt" );

    nn.learnRand1( subLearn, bigPenal, 3600 , 0, 1E-12, 0.0005 , 0.01 , rnd(), false, true, false );
    stdOut << "subLearn.size=" << subLearn.size() << " nn.sizeWeights=" << nn.sizeWeights() << " learn error=" << nn.error( subLearn , bigPenal ) << " class test=" << classify( test ,nn ) << "% class learn=" << classify( subLearn ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "nndef_out.txt" );

    nn.learnRand1( subLearn, bigPenal, 3600 , 0, 1E-12, 0.0001 , 0.001 , rnd(), true, true, false );
    stdOut << "subLearn.size=" << subLearn.size() << " nn.sizeWeights=" << nn.sizeWeights() << " learn error=" << nn.error( subLearn , bigPenal ) << " class test=" << classify( test ,nn ) << "% class learn=" << classify( subLearn ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "nndef_out.txt" );


    nn.momentum2( subLearn , CNNData(), 1800, 0, 0.001, 0.1, 1E-9, 0.8, CTVFlt(), bigPenal, 3, nullptr);
    stdOut << "subLearn.size=" << subLearn.size() << " nn.sizeWeights=" << nn.sizeWeights() << " learn error=" << nn.error( subLearn , bigPenal ) << " class test=" << classify( test ,nn ) << "% class learn=" << classify( subLearn ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "nndef_out.txt" );

    nn.toUniqueWeights();

    nn.momentum2( subLearn , CNNData(), 3600, 0, 0.001, 0.1, 1E-9, 0.8, CTVFlt(), bigPenal, 3, nullptr);
    stdOut << "subLearn.size=" << subLearn.size() << " nn.sizeWeights=" << nn.sizeWeights() << " learn error=" << nn.error( subLearn , bigPenal ) << " class test=" << classify( test ,nn ) << "% class learn=" << classify( subLearn ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "nndef_out.txt" );


}


//4..25..5 learn error=0.199088 test=78.4856% time=34699s
void experimentB() {
    QTextStream stdOut(stdout);

    const int learnSize = 80000;
    NNData learn, test;
    {
        FRnd rnd(1);
        CNNData data = NNData::mkData(false,0,4,0,3,5,"/home/m/tmp/test_data.csv",",");
        data.split( learn , learnSize, test, data.size()-learnSize, rnd() );
    }

//  const unsigned long long rndSeed = 2634215374;
    const unsigned long long rndSeed = std::random_device()();

    stdOut << "rndSeed=" << rndSeed << endl;
    FRnd rnd( rndSeed );

    NNData subLearn = learn;

    const time_t start = time(NULL);
    nnftyp bigPenal = 1E-12;

    NNNet nn;
    nn.read("nndef.txt" );
    stdOut << " learn error=" << nn.error( subLearn , bigPenal ) << " class test=" << classify( test ,nn ) << "% class learn=" << classify( subLearn ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;

    nn.toUniqueWeights();

    nn.randWeights( rnd , -1.0 , +1.0 );
    nn.randInputs(rnd);
    nn.randIdxW( rnd );

    stdOut << " learn error=" << nn.error( subLearn , bigPenal ) << " class test=" << classify( test ,nn ) << "% class learn=" << classify( subLearn ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "nndef_out.txt" );

//    nn.forceIdx(subLearn,bigPenal,1E4,true);
//    stdOut << " learn error=" << nn.error( subLearn , bigPenal ) << " class test=" << classify( test ,nn ) << "% class learn=" << classify( subLearn ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
//    nn.save( "nndef_out.txt" );

    for( int loop=1 ; true ; loop++ ) {
//        nn.learnRand1( subLearn, bigPenal, 120 , 0, 1E-12, 0.0005 , 0.01 , rnd(), true, true, true );
//        stdOut << " learn error=" << nn.error( subLearn , bigPenal ) << " class test=" << classify( test ,nn ) << "% class learn=" << classify( subLearn ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
//        nn.save( "nndef_out.txt" );

        nn.momentum2( subLearn , CNNData(), 1800, 0, 0.001, 0.1, 1E-9, 0.99, CTVFlt(), bigPenal, 5, nullptr);
        stdOut << " learn error=" << nn.error( subLearn , bigPenal ) << " class test=" << classify( test ,nn ) << "% class learn=" << classify( subLearn ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );

//        nn.toUniqueWeights();
//        subLearn = subLearn;
    }

}


int main(int argc, char *argv[])
{
    experimentA();
    return 0;
}

/*
subLearn.size=35000 nn.sizeWeights=71 learn error=0.251538 class test=73.28% class learn=73.7943% time=46243s
subLearn.size=80000 nn.sizeWeights=309 learn error=0.143869 class test=85.265% class learn=85.5888% time=36621s

35] 19 6 15 0.28951977

subLearn.size=10000 nn.sizeWeights=31 learn error=0.280226 class test=67.92% class learn=68.93% time=8886s

subLearn.size=80000 nn.sizeWeights=309 learn error=0.125549 class test=87.85% class learn=88.1287% time=36093s


subLearn.size=10000 nn.sizeWeights=31 learn error=0.239344 class test=73.94% class learn=75.21% time=6458s

learn error=0.137884 class test=86.125% class learn=86.5212% time=18048s

subLearn.size=80000 nn.sizeWeights=384 learn error=0.174126 class test=82.04% class learn=81.9663% time=29028s
*/
