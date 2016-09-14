package org.maltparser.core.symbol.hash;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.regex.Pattern;


import org.maltparser.core.exception.MaltChainedException;
import org.maltparser.core.helper.HashMap;
import org.maltparser.core.io.dataformat.ColumnDescription;
import org.maltparser.core.symbol.SymbolException;
import org.maltparser.core.symbol.SymbolTable;
import org.maltparser.core.symbol.nullvalue.InputNullValues;
import org.maltparser.core.symbol.nullvalue.NullValues;
import org.maltparser.core.symbol.nullvalue.OutputNullValues;
import org.maltparser.core.symbol.nullvalue.NullValues.NullValueId;


public final class HashSymbolTable implements SymbolTable {
	private final String name;
	private final Map<String, Integer> symbolCodeMap;
	private final SortedMap<Integer, String> codeSymbolMap;
	private final NullValues nullValues;
	private final int columnCategory;
	private int valueCounter;
	
	public HashSymbolTable(String name, int columnCategory, String nullValueStrategy) throws MaltChainedException {
		this.name = name;
		this.columnCategory = columnCategory;
		this.symbolCodeMap = new HashMap<String, Integer>();
		this.codeSymbolMap = new TreeMap<Integer, String>();
		if (columnCategory == ColumnDescription.INPUT) {
			this.nullValues = new InputNullValues(nullValueStrategy, this);
		} else if (columnCategory == ColumnDescription.DEPENDENCY_EDGE_LABEL) {
			this.nullValues = new OutputNullValues(nullValueStrategy, this);
		} else {
			this.nullValues = new InputNullValues(nullValueStrategy, this);
		}
		this.valueCounter = nullValues.getNextCode();
	}
	
	public HashSymbolTable(String name) { 
		this.name = name;
		this.columnCategory = -1;
		this.symbolCodeMap = new HashMap<String, Integer>();
		this.codeSymbolMap = new TreeMap<Integer, String>();
		this.nullValues = new InputNullValues("one", this);
		this.valueCounter = 1;
	}
	
	public int addSymbol(String symbol) throws MaltChainedException {
		if (nullValues == null || !nullValues.isNullValue(symbol)) {
			if (symbol == null || symbol.length() == 0) {
				throw new SymbolException("Symbol table error: empty string cannot be added to the symbol table");
			}
	
			if (!symbolCodeMap.containsKey(symbol)) {
				int code = valueCounter;
				symbolCodeMap.put(symbol, code);
				codeSymbolMap.put(code, symbol);
				valueCounter++;
				return code;
			} else {
				return symbolCodeMap.get(symbol);
			}
		} else {
			return nullValues.symbolToCode(symbol);
		}

	}
	
	public String getSymbolCodeToString(int code) throws MaltChainedException {
		if (code >= 0) {
			if (nullValues == null || !nullValues.isNullValue(code)) {
				if (codeSymbolMap.containsKey(code)) {
					return codeSymbolMap.get(code);
				} else {
					return null;
				}
			} else {
				return nullValues.codeToSymbol(code);
			}
		} else {
			throw new SymbolException("The symbol code '"+code+"' cannot be found in the symbol table. ");
		}
	}
	
	public int getSymbolStringToCode(String symbol) throws MaltChainedException {
		if (symbol != null) {
			if (nullValues == null || !nullValues.isNullValue(symbol)) {
				if (symbolCodeMap.containsKey(symbol)) {
					return symbolCodeMap.get(symbol);
				} else {
					return -1;
				}
			} else {
				return nullValues.symbolToCode(symbol);
			}
		} else {
			throw new SymbolException("The symbol code '"+symbol+"' cannot be found in the symbol table. ");
		}
	}
	
	public String printSymbolTable() throws MaltChainedException {
		StringBuilder sb = new StringBuilder();
		for (Integer code : codeSymbolMap.keySet()) {
			sb.append(code+"\t"+codeSymbolMap.get(code)+"\n");
		}
		return sb.toString();
	}
	
	public void saveHeader(BufferedWriter out) throws MaltChainedException  {
		try {
			out.append('\t');
			out.append(getName());
			out.append('\t');
			out.append(Integer.toString(getColumnCategory()));
			out.append('\t');
			out.append(getNullValueStrategy());
			out.append('\n');
		} catch (IOException e) {
			throw new SymbolException("Could not save the symbol table. ", e);
		}
	}
	
	public int getColumnCategory() {
		return columnCategory;
	}
	
	public String getNullValueStrategy() {
		if (nullValues == null) {
			return null;
		}
		return nullValues.getNullValueStrategy();
	}
	
	public int size() {
		return symbolCodeMap.size();
	}
	
	public void save(BufferedWriter out) throws MaltChainedException  {
		try {
			out.write(name);
			out.write('\n');
			for (Integer code : codeSymbolMap.keySet()) {
				out.write(Integer.toString(code));
				out.write('\t');
				out.write(codeSymbolMap.get(code));
				out.write('\n');
			}
			out.write('\n');
		} catch (IOException e) {
			throw new SymbolException("Could not save the symbol table. ", e);
		}
	}
	
	public void load(BufferedReader in) throws MaltChainedException {		
		int max = 0;
		String fileLine;
		Pattern splitPattern = Pattern.compile("\t");
		try {
			while ((fileLine = in.readLine()) != null) {
				if (fileLine.length() == 0) {
					valueCounter = max+1;
					break;
				}
				String[] items = splitPattern.split(fileLine);
				int code = Integer.parseInt(items[0]);
				symbolCodeMap.put(items[1], code);
				codeSymbolMap.put(code, items[1]);
				if (max < code) {
					max = code;
				}
			}
		} catch (NumberFormatException e) {
			throw new SymbolException("The symbol table file (.sym) contains a non-integer value in the first column. ", e);
		} catch (IOException e) {
			throw new SymbolException("Could not load the symbol table. ", e);
		}
	}
	
	public String getName() {
		return name;
	}

	public int getValueCounter() {
		return valueCounter;
	}

	public int getNullValueCode(NullValueId nullValueIdentifier) throws MaltChainedException {
		if (nullValues == null) {
			throw new SymbolException("The symbol table does not have any null-values. ");
		}
		return nullValues.nullvalueToCode(nullValueIdentifier);
	}
	
	public String getNullValueSymbol(NullValueId nullValueIdentifier) throws MaltChainedException {
		if (nullValues == null) {
			throw new SymbolException("The symbol table does not have any null-values. ");
		}
		return nullValues.nullvalueToSymbol(nullValueIdentifier);
	}
	
	public boolean isNullValue(String symbol) throws MaltChainedException {
		if (nullValues != null) {
			return nullValues.isNullValue(symbol);
		} 
		return false;
	}
	
	public boolean isNullValue(int code) throws MaltChainedException {
		if (nullValues != null) {
			return nullValues.isNullValue(code);
		} 
		return false;
	}
	
	public Set<Integer> getCodes() {
		return codeSymbolMap.keySet();
	}
	
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		final HashSymbolTable other = (HashSymbolTable)obj;
		return ((name == null) ? other.name == null : name.equals(other.name));
	}

	public int hashCode() {
		return 217 + (null == name ? 0 : name.hashCode());
	}
	
	public String toString() {
		final StringBuilder sb = new StringBuilder();
		sb.append(name);
		sb.append(' ');
		sb.append(valueCounter);
		return sb.toString();
	}
}
