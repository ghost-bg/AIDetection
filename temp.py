import schemdraw
import schemdraw.logic as logic
import schemdraw.elements as elm

with schemdraw.Drawing() as d:
    # Inverters for B', C', D'
    d += (B_inv := logic.Not().label("B'", 'right').anchor('in'))
    d += elm.Line().left().at(B_inv.in_).label('B', 'left')

    d += (C_inv := logic.Not().label("C'", 'right').at(B_inv.in_).down(2).anchor('in'))
    d += elm.Line().left().at(C_inv.in_).label('C', 'left')

    d += (D_inv := logic.Not().label("D'", 'right').at(C_inv.in_).down(2).anchor('in'))
    d += elm.Line().left().at(D_inv.in_).label('D', 'left')

    # First AND gate chain for B'C'D'
    # AND gate combining B' and C'
    d += (and1 := logic.And().at((2, -1)).anchor('in1'))
    d += elm.Line().at(B_inv.out).to(and1.in1)
    d += elm.Line().at(C_inv.out).to(and1.in2)

    # AND gate combining (B'·C') and D'
    d += (and2 := logic.And().at((4, -1)).anchor('in1'))
    d += elm.Line().at(and1.out).to(and2.in1)
    d += elm.Line().at(D_inv.out).to(and2.in2)
    # Output label
    d += elm.Line().at(and2.out).right(0.5).label("$B'C'D'$", 'right')

    # XOR gate for A ⊕ C
    d += (xor1 := logic.Xor().at((2, -5)).anchor('in1'))
    d += elm.Line().left(1).at(xor1.in1).label('A', 'left')
    d += elm.Line().left(1).at(xor1.in2).label('C', 'left')

    # AND gate combining B and D
    d += (and3 := logic.And().at((2, -7)).anchor('in1'))
    d += elm.Line().left(1).at(and3.in1).label('B', 'left')
    d += elm.Line().left(1).at(and3.in2).label('D', 'left')

    # AND gate combining BD and (A ⊕ C)
    d += (and4 := logic.And().at((4, -6)).anchor('in1'))
    d += elm.Line().at(xor1.out).to(and4.in1)
    d += elm.Line().at(and3.out).to(and4.in2)
    # Output label
    d += elm.Line().at(and4.out).right(0.5).label("$BD(A \\oplus C)$", 'right')

    # OR gate to obtain final output R
    d += (or1 := logic.Or().at((6, -3.5)).anchor('in1'))
    d += elm.Line().at(and2.out).to(or1.in1)
    d += elm.Line().at(and4.out).to(or1.in2)
    d += elm.Line().at(or1.out).right(1).label('R', 'right')

    d.draw()
