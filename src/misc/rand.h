#pragma once

#include <random>
#include "src/defs.h"
#include "critical.h"

class FRnd : public std::ranlux48 {
private:
    std::uniform_real_distribution<ftyp> fdist;
public:
    FRnd( cutyp seed ):std::ranlux48( seed ),fdist(0.0,1.0) {
    }

    ftyp getF() {
        return fdist(*this);
    }

    utyp getU() {
        return (utyp) (*this)();
    }

    ultyp getUL() {
        return (((ultyp)getU()) << 32) | ((ultyp)getU());
    }

    ityp  getI() {
        return (ityp)(getU() & 0x7FFFFFFF);
    }

    ityp  getI(cityp max) {
        return getI() % max;
    }

    ityp  getI(cityp min, cityp max) {
        return getI() % ( max - min + 1 ) + min;
    }

    ftyp getF(cftyp min, cftyp max) {
        return getF() * ( max - min ) + min;
    }

    bool getB() {
        return getU() % 512 < 256;
    }

    void chaos(ftyp &value, cftyp min, cftyp max, cftyp strength) {
        EXCP( value <= max );
        EXCP( value >= min );
        if( getB() ) {
            value += getF( 0 , max-value ) * strength;
        } else {
            value -= getF( 0 , value-min ) * strength;
        }
    }

    void chaos(lftyp &value, cftyp min, cftyp max, cftyp strength) {
        EXCP( value <= max );
        EXCP( value >= min );
        if( getB() ) {
            value += getF( 0 , max-value ) * strength;
        } else {
            value -= getF( 0 , value-min ) * strength;
        }
    }

    void chaos(__float128 &value, cftyp min, cftyp max, cftyp strength) {
        EXCP( value <= max );
        EXCP( value >= min );
        if( getB() ) {
            value += getF( 0 , max-value ) * strength;
        } else {
            value -= getF( 0 , value-min ) * strength;
        }
    }

};
