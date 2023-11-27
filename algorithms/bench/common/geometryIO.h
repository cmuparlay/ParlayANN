// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once
#include "../parlay/parallel.h"
#include "../parlay/primitives.h"
#include "geometry.h"
#include "IO.h"

//using namespace geometry;
using namespace benchIO;

  template <class coord>
  inline int xToStringLen(point2d<coord> a) { 
    return xToStringLen(a.x) + xToStringLen(a.y) + 1;
  }

  template <class coord>
  inline void xToString(char* s, point2d<coord> a) { 
    int l = xToStringLen(a.x);
    xToString(s, a.x);
    s[l] = ' ';
    xToString(s+l+1, a.y);
  }

  template <class coord>
  inline int xToStringLen(point3d<coord> a) { 
    return xToStringLen(a.x) + xToStringLen(a.y) + xToStringLen(a.z) + 2;
  }

  template <class coord>
  inline void xToString(char* s, point3d<coord> a) { 
    int lx = xToStringLen(a.x);
    int ly = xToStringLen(a.y);
    xToString(s, a.x);
    s[lx] = ' ';
    xToString(s+lx+1, a.y);
    s[lx+ly+1] = ' ';
    xToString(s+lx+ly+2, a.z);
  }

  // inline int xToStringLen(tri a) { 
  //   return xToStringLen(a[0]) + xToStringLen(a[1]) + xToStringLen(a[2]) + 2;
  // }

  // inline void xToString(char* s, tri a) { 
  //   int lx = xToStringLen(a[0]);
  //   int ly = xToStringLen(a[1]);
  //   xToString(s, a[0]);
  //   s[lx] = ' ';
  //   xToString(s+lx+1, a[1]);
  //   s[lx+ly+1] = ' ';
  //   xToString(s+lx+ly+2, a[2]);
  // }

namespace benchIO {
  using namespace std;

  string HeaderPoint2d = "pbbs_sequencePoint2d";
  string HeaderPoint3d = "pbbs_sequencePoint3d";
  string HeaderTriangles = "pbbs_triangles";

  template <class Point>
    int writePointsToFile(parlay::sequence<Point> const &P, char const *fname) {
    string Header = (Point::dim == 2) ? HeaderPoint2d : HeaderPoint3d;
    int r = writeSeqToFile(Header, P, fname);
    return r;
  }

  template <class Point, class Seq>
  parlay::sequence<Point> parsePoints(Seq W) {
    using coord = typename Point::coord;
    int d = Point::dim;
    size_t n = W.size()/d;
    auto a = parlay::tabulate(d * n, [&] (size_t i) -> coord {
	return atof(W[i]);});
    auto points = parlay::tabulate(n, [&] (size_t i) -> Point {
	return Point(a.cut(d*i,d*(i + 1)));});
    return points;
  }

  template <class Point>
  parlay::sequence<Point> readPointsFromFile(char const *fname) {
    parlay::sequence<char> S = readStringFromFile(fname);
    parlay::sequence<char*> W = stringToWords(S);
    int d = Point::dim;
    if (W.size() == 0 || W[0] != (d == 2 ? HeaderPoint2d : HeaderPoint3d)) {
      cout << "readPointsFromFile wrong file type" << endl;
      abort();
    }
    return parsePoints<Point>(W.cut(1,W.size()));
  }

  // triangles<point2d> readTrianglesFromFileNodeEle(char const *fname) {
  //   string nfilename(fname);
  //   _seq<char> S = readStringFromFile((char*)nfilename.append(".node").c_str());
  //   words W = stringToWords(S.A, S.n);
  //   triangles<point2d> Tr;
  //   Tr.numPoints = atol(W.Strings[0]);
  //   if (W.m < 4*Tr.numPoints + 4) {
  //     cout << "readStringFromFileNodeEle inconsistent length" << endl;
  //     abort();
  //   }

  //   Tr.P = newA(point2d, Tr.numPoints);
  //   for(intT i=0; i < Tr.numPoints; i++) 
  //     Tr.P[i] = point2d(atof(W.Strings[4*i+5]), atof(W.Strings[4*i+6]));

  //   string efilename(fname);
  //   _seq<char> SN = readStringFromFile((char*)efilename.append(".ele").c_str());
  //   words WE = stringToWords(SN.A, SN.n);
  //   Tr.numTriangles = atol(WE.Strings[0]);
  //   if (WE.m < 4*Tr.numTriangles + 3) {
  //     cout << "readStringFromFileNodeEle inconsistent length" << endl;
  //     abort();
  //   }

  //   Tr.T = newA(triangle, Tr.numTriangles);
  //   for (long i=0; i < Tr.numTriangles; i++)
  //     for (int j=0; j < 3; j++)
  // 	Tr.T[i].C[j] = atol(WE.Strings[4*i + 4 + j]);

  //   return Tr;
  // }

  template <class pointT>
  triangles<pointT> readTrianglesFromFile(char const *fname, int offset) {
    int d = pointT::dim;
    parlay::sequence<char> S = readStringFromFile(fname);
    parlay::sequence<char*> W = stringToWords(S);
    if (W[0] != HeaderTriangles) {
      cout << "readTrianglesFromFile wrong file type" << endl;
      abort();
    }

    int headerSize = 3;
    size_t n = atol(W[1]);
    size_t m = atol(W[2]);
    if (W.size() != headerSize + 3 * m + d * n) {
      cout << "readTrianglesFromFile inconsistent length" << endl;
      abort();
    }

    auto pts_slice = W.cut(headerSize, headerSize + d * n);
    auto tri_slice = W.cut(headerSize + d * n, W.size());
    parlay::sequence<pointT> Pts = parsePoints<pointT>(pts_slice);
    auto Tri = parlay::tabulate(m, [&] (size_t i ) -> tri {
				     return {(int) atol(tri_slice[3*i])-offset,
					     (int) atol(tri_slice[3*i+1])-offset,
					     (int) atol(tri_slice[3*i+2])-offset};});
    return triangles<pointT>(Pts,Tri);
  }

  template <class pointT>
  int writeTrianglesToFile(triangles<pointT> Tr, char* fileName) {
    ofstream file (fileName, ios::binary);
    if (!file.is_open()) {
      std::cout << "Unable to open file: " << fileName << std::endl;
      return 1;
    }
    file << HeaderTriangles << endl;
    file << Tr.numPoints() << endl; 
    file << Tr.numTriangles() << endl; 
    writeSeqToStream(file, Tr.P);
    //writeSeqToStream(file, Tr.T);
    auto A = parlay::tabulate(3*Tr.numTriangles(), [&] (size_t i) -> int {
      						     return (Tr.T[i/3])[i%3];});
    writeSeqToStream(file, A);
    file.close();
    return 0;
  }

};

