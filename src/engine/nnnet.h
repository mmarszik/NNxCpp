#pragma once
#include <QString>
#include "src/defs.h"
#include "src/misc/rand.h"
#include "data.h"

namespace NsNet {

class NNNeuron;  // Neuron
class NNNet;     // Sieć

typedef const NNNeuron CNNNeuron;

// Typ aktywacji
enum NNActv {
    NNA_LIN,    // Liniowa
    NNA_MLIN,   // Liniowa łamana
    NNA_SUNI,   // Sigmoidalna unipolarna
    NNA_SBIP,   // Sigmoidalna bipolarna
    NNA_SBIP_L, // Sigmoidalna bipolarna + liniowa
    NNA_RELU,   // Softmax
    NNA_NNAME   // No-name
};

// Wektor neuronów.
typedef NNVec<NNNeuron> TVNer;
typedef const TVNer     CTVNer;

// Neuron
class NNNeuron {    
friend class NNNet;
    NNActv actv;        // Typ neuronu.
    nnityp conv;        // Indeks splotu do którego należy ten neuron. Indeks ujemny oznacza, że w splocie jest tylko ten jeden neuron

    TVInt  idx_i;       // Indeksy wejść.
    nnityp min_i;       // Minimalny indeks wejścia w optymalizacji.
    nnityp max_i;       // Maksymalny indeks wejścia w optymalizacji.

    TVInt  idx_w;       // Indeksy wag. Wszystkie połączenia w jednym splocie muszą mieć takie same wagi.
    nnityp min_w;       // Minimalny indeks wag.
    nnityp max_w;       // Maksymalny indeks wag.

    TVInt  idx_o;       // Indeksy wyjść, przydatne do obliczania pochodnej.
    nnityp min_s;       // Minimalny rozmiar wejścia (i tym samym wag)
    nnityp max_s;       // Maksymalny rozmiar wejścia (i tym samym wag)
public:
    NNNeuron(){}
    bool isConvolution() const { return conv != -1; }

};

// Sieć neuronowa
class NNNet {
    nnityp size_i;        // Rozmiar wejścia.
    nnityp size_o;        // Rozmiar wyjścia.
    TVFlt weights;        // Wektor wag.
    nnftyp gmin_w,gmax_w; // Ogólne ograczenia wag.
    TVFlt min_w;          // Minimalne ograniczenia wag.
    TVFlt max_w;          // Maksymalne ograniczenia wag.
    TVNer neurons;        // Wektor neuronów.
    TVIInt convs;         // Wektor splotów; gdzie splot stanowi wektor intów.
    nnftyp weights_penal; // Kara za duże wartości wag.
    nnftyp signal_penal;  // Kara za duże wartosci sygnałów.
    nnityp rnd_seed;      // Zarodek liczb losowych.

private:
    bool chaosWeight(FRnd &rnd, cftyp strength );
    bool chaosIdxWeight( FRnd &rnd );
    bool chaosIdxInput( FRnd &rnd );
    bool chaosSizeUp( FRnd &rnd );

public:
    void chaosWeights(nncityp rndSeed, cftyp strength, nncityp loops=1);

public:
    NNNet() {}

    void compute(CTVFlt &inp, TVFlt &obuf) const;

    nnftyp error(CTVFlt &inp , CTVFlt &out , TVFlt &obuf) const;
    nnftyp error( CNNRecord &rec , TVFlt &obuf ) const;
    nnftyp error(CNNData &data , nncftyp bigpenal=0, nnityp start=-1, nnityp end=-1) const;

    nnityp sizeBuf() const { return size_i + neurons.size(); }
    nnityp offsetOut() const { return sizeBuf() - size_o; }
    void compute(nncftyp *inp , nnftyp obuf[] ) const;
    TVFlt compute( CTVFlt &inp ) const;

    nnftyp simpClass( CNNData &data ) const;

    bool chaos(FRnd &rnd, cftyp strength, const bool weights=true, const bool idx_weights=true, const bool idx_input=true);
    void read(const QString& path);
    void save(const QString& path) const;
    void mkBackConnect();

    void gradient( CTVFlt &inp , CTVFlt &out , TVFlt &obuf , TVFlt &ibuf , TVFlt &grad  ) const;
    void gradientN( CTVFlt &inp , CTVFlt &out , TVFlt &obuf , TVFlt &grad  );
    TVFlt gradientN( CNNData &data );

    nnityp sizeWeights() const { return weights.size(); }
    nnityp sizeNeurons() const { return neurons.size(); }
    TVFlt gradient(CNNData &data , nncftyp bigpenal=0) const;
    TVFlt subGradient(CNNData &data , CTVInt &sub ) const;
    void rawDescent(CNNData &data , nncityp loops , nncftyp step , CTVFlt &tolearn);
    void grDescent( CNNData &data , nncityp loops , nncftyp step , CTVFlt &tolearn, nncftyp min_error=0.000001, nnityp subloops=40 , nnityp show=1 );
    void rawMomentum( CNNData &data , nncityp loops , nncftyp step , TVFlt &p , nncftyp mom,  nncftyp bigpenal, CTVFlt &tolearn);
    void momentum(CNNData &data , nncityp loops , nncftyp step , nncftyp mom, CTVFlt &tolearn, nncftyp min_error, nncftyp bigpenal, nnityp subloops , nnityp show=1);
    void bfgs( CNNData &data , CTVInt &sub , nncityp loops, nncftyp step );
    void learnRand1(CNNData &data , nncftyp bigPenal, nncityp maxTime , nncityp maxnoti, nncftyp mindesc,  nncftyp min_str, nncftyp max_str, nncityp seed, const bool weights, const bool idx_weights, const bool idx_input );
    void learnRand2(NNData  &data,
        CTVInt  &flods, nncityp multi,
        nncityp loops,
        nncityp subLoops,
        nncityp notInc, nncftyp bigPenal,
        nncftyp minDesc,
        nncftyp minStrength,
        nncftyp maxStrength,
        nncityp seed,
        nncftyp pBack,
        const bool weights,
        const bool idx_weights,
        const bool idx_input
    );

    int forceIdx( CNNData &data , nncftyp bigPenal, const time_t maxTime, const int maxLoops, const bool show );
    int forceIdx2(
        CNNData      &data ,
        nncftyp      bigPenal,
        const time_t maxTime,
        nncityp      maxLoops,
        const bool   show
    );
    void randIdxW(FRnd &rnd , nncftyp min=0 , nncftyp max=0 , nncityp maxTry=1000);
    void randWeights(FRnd &rnd , nncftyp min=0, nncftyp max=0);
    void randInputs(FRnd &rnd);
    void mkNewData( const QString& path, CNNData &data ) const;
    void momentum2(CNNData &learn ,
        CNNData &test ,
        const time_t maxTime ,
        nncityp maxFailsTest,
        nnftyp step,
        nncftyp maxStep,
        nncftyp minStep,
        nncftyp mom,
        CTVFlt &tolearn,
        nncftyp bigpenal,
        nnityp subLoops ,
        NNNet *const best = nullptr
    );
    void learnRndIdx(
        CNNData &data ,            //MM: dane uczące
        const time_t maxTime ,     //MM: maksymalna ilość czasu
        nncityp maxNotLearn ,      //MM: maksymalna ilość iteracji bez minimalnego spadku błędu
        nncftyp minError ,         //MM: wartość minimalnego spadek błędu
        nncftyp bigpenal ,         //MM: kara za duże wagi
        nncityp rndSeed ,          //MM: zarodek liczb losowych
        nncftyp pBack              //MM: prawdopodobieństwo powrotu do lepszego rozwiązania
    );
    int doubleRndIdxWeightForce(
        CNNData      &data ,
        const time_t maxTime,
        nncityp      maxLoops,
        nncityp      maxNotIncrese,
        nncityp      rndSeed,
        const bool   show = true
    );
    void appendWeight(nncftyp weight, nncftyp min_w, nncftyp max_w);
    void extend(nncftyp weight, nncftyp min_w, nncftyp max_w);
    int subForceIdx(NNData &data, nncftyp bigPenal, CTVInt &parts, nncityp maxTime, nncityp maxFails, nncityp rndSeed, const bool show=true);
    void toUniqueWeights(nncityp skip=0 );
    void setMinWeights(nncftyp min);
    void setMaxWeights(nncftyp max);

    void annealing(CNNData &data, nncftyp bigPenal, nncityp maxTime, nncftyp minStrength, nncftyp maxStrength, nncftyp pBack, nnityp rndSeed);
    int forcePairIdx(CNNData &data, nncftyp bigPenal, nncityp loops, const bool show, nncityp rndSeed);
    void learnRand3(CNNData &learn, CNNData &test, const time_t maxTime, nncityp maxFailsTest, nncftyp maxStep, nncftyp minStep, nncftyp bigPenal, nncityp rndSeed, NNNet *const best=NULL, const bool fullVerb=false);

    void multiInit(CNNData &data,
        const time_t maxTime,
        nncityp maxLoops,
        nncityp minWeight,
        nncityp maxWeight,
        const bool rndWeight,
        const bool rndIdxWeight,
        const bool rndInput,
        nncftyp bigPenal = 1E-9,
        nncityp rndSeed = 1,
        nncityp bestSize = 100 ,
        const bool show = true
    );

};




}
