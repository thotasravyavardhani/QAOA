import React from 'react';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';

export function MaxCutFormula() {
  return (
    <Card className="mb-4 bg-gradient-to-r from-blue-50 to-indigo-50 border-2 border-blue-200">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg text-blue-900">📐 Max-Cut Problem Formulation</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="bg-white p-4 rounded-lg shadow-sm">
          <p className="text-sm text-gray-700 mb-2 font-semibold">Objective Function:</p>
          <div className="overflow-x-auto">
            <BlockMath math="C(z) = \sum_{(i,j) \in E} w_{ij} \cdot \frac{1 - z_i z_j}{2}" />
          </div>
          <p className="text-xs text-gray-600 mt-2">
            where <InlineMath math="z_i \in \{-1, +1\}" /> represents the partition assignment of vertex i
          </p>
        </div>
        
        <div className="bg-white p-4 rounded-lg shadow-sm">
          <p className="text-sm text-gray-700 mb-2 font-semibold">QAOA Ansatz:</p>
          <div className="overflow-x-auto">
            <BlockMath math="|\psi(\boldsymbol{\gamma}, \boldsymbol{\beta})\rangle = U_M(\beta_p)U_C(\gamma_p) \cdots U_M(\beta_1)U_C(\gamma_1)|+\rangle^{\otimes n}" />
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-purple-50 p-3 rounded-lg border border-purple-200">
            <p className="text-xs text-purple-900 font-semibold mb-1">Cost Hamiltonian:</p>
            <div className="text-sm">
              <InlineMath math="U_C(\gamma) = e^{-i\gamma H_C}" />
            </div>
          </div>
          <div className="bg-orange-50 p-3 rounded-lg border border-orange-200">
            <p className="text-xs text-orange-900 font-semibold mb-1">Mixer Hamiltonian:</p>
            <div className="text-sm">
              <InlineMath math="U_M(\beta) = e^{-i\beta H_M}" />
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function TSPFormula() {
  return (
    <Card className="mb-4 bg-gradient-to-r from-green-50 to-teal-50 border-2 border-green-200">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg text-green-900">📐 TSP Problem Formulation</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="bg-white p-4 rounded-lg shadow-sm">
          <p className="text-sm text-gray-700 mb-2 font-semibold">Objective Function:</p>
          <div className="overflow-x-auto">
            <BlockMath math="C(x) = \sum_{i=0}^{n-1} \sum_{j=0}^{n-1} d_{ij} \sum_{t=0}^{n-1} x_{it} x_{j,(t+1)\bmod n}" />
          </div>
          <p className="text-xs text-gray-600 mt-2">
            where <InlineMath math="x_{it} = 1" /> if city i is visited at time t, <InlineMath math="d_{ij}" /> is distance
          </p>
        </div>
        
        <div className="bg-white p-4 rounded-lg shadow-sm">
          <p className="text-sm text-gray-700 mb-2 font-semibold">Constraints (via Penalty):</p>
          <div className="overflow-x-auto text-sm space-y-1">
            <BlockMath math="H_{penalty} = A\left(\sum_{i} (1-\sum_t x_{it})^2 + \sum_{t} (1-\sum_i x_{it})^2\right)" />
          </div>
          <p className="text-xs text-gray-600 mt-2">
            Ensures each city visited exactly once and each time step has one city
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

export function VRPFormula() {
  return (
    <Card className="mb-4 bg-gradient-to-r from-amber-50 to-orange-50 border-2 border-amber-200">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg text-amber-900">📐 VRP Problem Formulation</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="bg-white p-4 rounded-lg shadow-sm">
          <p className="text-sm text-gray-700 mb-2 font-semibold">Objective Function:</p>
          <div className="overflow-x-auto">
            <BlockMath math="C = \sum_{k=1}^{K} \sum_{i=0}^{n} \sum_{j=0}^{n} d_{ij} \cdot x_{ijk}" />
          </div>
          <p className="text-xs text-gray-600 mt-2">
            Minimize total distance for K vehicles serving n customers
          </p>
        </div>
        
        <div className="bg-white p-4 rounded-lg shadow-sm">
          <p className="text-sm text-gray-700 mb-2 font-semibold">Capacity Constraint:</p>
          <div className="overflow-x-auto">
            <BlockMath math="\sum_{i=1}^{n} q_i \cdot x_{ik} \leq Q_k, \quad \forall k" />
          </div>
          <p className="text-xs text-gray-600 mt-2">
            where <InlineMath math="q_i" /> is demand of customer i, <InlineMath math="Q_k" /> is vehicle capacity
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

export function GraphColoringFormula() {
  return (
    <Card className="mb-4 bg-gradient-to-r from-pink-50 to-purple-50 border-2 border-pink-200">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg text-pink-900">📐 Graph Coloring Problem Formulation</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="bg-white p-4 rounded-lg shadow-sm">
          <p className="text-sm text-gray-700 mb-2 font-semibold">Objective Function (Minimize Conflicts):</p>
          <div className="overflow-x-auto">
            <BlockMath math="C = \sum_{(i,j) \in E} \sum_{c=0}^{k-1} x_{ic} \cdot x_{jc}" />
          </div>
          <p className="text-xs text-gray-600 mt-2">
            where <InlineMath math="x_{ic} = 1" /> if vertex i has color c (one-hot encoding)
          </p>
        </div>
        
        <div className="bg-white p-4 rounded-lg shadow-sm">
          <p className="text-sm text-gray-700 mb-2 font-semibold">One-Hot Constraint:</p>
          <div className="overflow-x-auto">
            <BlockMath math="H_{constraint} = A\sum_{i=0}^{n-1} \left(1 - \sum_{c=0}^{k-1} x_{ic}\right)^2" />
          </div>
          <p className="text-xs text-gray-600 mt-2">
            Each vertex must have exactly one color
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
