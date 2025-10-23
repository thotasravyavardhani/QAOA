import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Construction } from 'lucide-react';

export default function GraphColoringSolver() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Graph Coloring Problem</CardTitle>
        <CardDescription>Assign colors such that no adjacent vertices share the same color</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="text-center py-12 text-gray-400">
          <Construction className="w-16 h-16 mx-auto mb-4" />
          <p>Graph Coloring Solver - Similar to Max-Cut implementation</p>
          <p className="text-sm mt-2">Configure graph and number of colors</p>
        </div>
      </CardContent>
    </Card>
  );
}
