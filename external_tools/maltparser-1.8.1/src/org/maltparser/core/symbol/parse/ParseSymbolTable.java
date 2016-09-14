package org.maltparser.core.symbol.parse;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.util.Map;

import org.maltparser.core.exception.MaltChainedException;
import org.maltparser.core.helper.HashMap;
import org.maltparser.core.symbol.SymbolException;
import org.maltparser.core.symbol.SymbolTable;
import org.maltparser.core.symbol.SymbolTableHandler;

import org.maltparser.core.symbol.nullvalue.NullValues.NullValueId;


public class ParseSymbolTable implements SymbolTable {
	private final String name;
	private final SymbolTable parentSymbolTable;
    
	/** Special treatment during parsing */
	private final Map<String, Integer> symbolCodeMap;
	private final Map<Integer, String> codeSymbolMap;
	private int valueCounter;
    
	public ParseSymbolTable(String name, int columnCategory, String nullValueStrategy, SymbolTableHandler parentSymbolTableHandler) throws MaltChainedException {
		this.name = name;
		this.parentSymbolTable = parentSymbolTableHandler.addSymbolTable(name, columnCategory, nullValueStrategy);
		this.symbolCodeMap = new HashMap<String, Integer>();
		this.codeSymbolMap = new HashMap<Integer, String>();
		this.valueCounter = -1;
	}
	
	public ParseSymbolTable(String name, SymbolTable parentTable, SymbolTableHandler parentSymbolTableHandler) throws MaltChainedException {
		this.name = name;
		this.parentSymbolTable = parentSymbolTableHandler.addSymbolTable(name, parentTable);
		this.symbolCodeMap = new HashMap<String, Integer>();
		this.codeSymbolMap = new HashMap<Integer, String>();
		this.valueCounter = -1;
	}
	
	public ParseSymbolTable(String name, SymbolTableHandler parentSymbolTableHandler) throws MaltChainedException {
		this.name = name;
		this.parentSymbolTable = parentSymbolTableHandler.addSymbolTable(name);
		this.symbolCodeMap = new HashMap<String, Integer>();
		this.codeSymbolMap = new HashMap<Integer, String>();
		this.valueCounter = -1;
	}
	
	public int addSymbol(String symbol) throws MaltChainedException {
		if (!parentSymbolTable.isNullValue(symbol)) {
			if (symbol == null || symbol.length() == 0) {
				throw new SymbolException("Symbol table error: empty string cannot be added to the symbol table");
			}

			int code = parentSymbolTable.getSymbolStringToCode(symbol); 
			if (code > -1) {
				return code;
			}
			if (!symbolCodeMap.containsKey(symbol)) {
//				System.out.println("!symbolCodeMap.containsKey(symbol) : " + this.getName() + ": " + symbol.toString());
				if (valueCounter == -1) {
					valueCounter = parentSymbolTable.getValueCounter() + 1;
				} else {
					valueCounter++;
				}
				symbolCodeMap.put(symbol, valueCounter);
				codeSymbolMap.put(valueCounter, symbol);
				return valueCounter;
			} else {
				return symbolCodeMap.get(symbol);
			}
		} else {
			return parentSymbolTable.getSymbolStringToCode(symbol);
		}
	}
	
//	public int addSymbol(StringBuilder symbol) throws MaltChainedException {
//		return addSymbol(symbol.toString());
//	}
	
	public String getSymbolCodeToString(int code) throws MaltChainedException {
		if (code >= 0) {
			if (!parentSymbolTable.isNullValue(code)) {
				String symbol = parentSymbolTable.getSymbolCodeToString(code); 
				if (symbol != null) {
					return symbol;
				} else {
					if (!codeSymbolMap.containsKey(code)) {
						throw new SymbolException("The symbol code '"+code+"' cannot be found in the symbol table. ");
					}
					return codeSymbolMap.get(code);
				}
			} else {
				return parentSymbolTable.getSymbolCodeToString(code);
			}
		} else {
			throw new SymbolException("The symbol code '"+code+"' cannot be found in the symbol table. ");
		}
	}
	
	public int getSymbolStringToCode(String symbol) throws MaltChainedException {
		if (symbol != null) {
			if (!parentSymbolTable.isNullValue(symbol)) {
				int code = parentSymbolTable.getSymbolStringToCode(symbol); 
				if (code > -1) {
					return code;
				}
				if (!symbolCodeMap.containsKey(symbol)) {
					throw new SymbolException("Could not find the symbol '"+symbol+"' in the symbol table. "); 
				}
				Integer item = symbolCodeMap.get(symbol);
				if (item == null) {
					throw new SymbolException("Could not find the symbol '"+symbol+"' in the symbol table. "); 
				} 
				return item.intValue();
			} else {
				return parentSymbolTable.getSymbolStringToCode(symbol);
			}
		} else {
			throw new SymbolException("The symbol code '"+symbol+"' cannot be found in the symbol table. ");
		}
	}

	public void clearTmpStorage() {
		symbolCodeMap.clear();
		codeSymbolMap.clear();
		valueCounter = -1;
	}
	
//	public String getNullValueStrategy() {
//		return parentSymbolTable.getNullValueStrategy();
//	}
//	
//	
//	public int getColumnCategory() {
//		return parentSymbolTable.getColumnCategory();
//	}
	
	public String printSymbolTable() throws MaltChainedException {
		return parentSymbolTable.printSymbolTable();
	}
	
//	public void saveHeader(BufferedWriter out) throws MaltChainedException  {
//		parentSymbolTable.saveHeader(out);
//	}
	
	public int size() {
		return parentSymbolTable.size();
	}
	
	
	public void save(BufferedWriter out) throws MaltChainedException  {
		parentSymbolTable.save(out);
	}
	
	public void load(BufferedReader in) throws MaltChainedException {
		parentSymbolTable.load(in);
	}
	
	public String getName() {
		return name;
	}

	public int getValueCounter() {
		return parentSymbolTable.getValueCounter();
	}

	
	public int getNullValueCode(NullValueId nullValueIdentifier) throws MaltChainedException {
		return parentSymbolTable.getNullValueCode(nullValueIdentifier);
	}
	
	public String getNullValueSymbol(NullValueId nullValueIdentifier) throws MaltChainedException {
		return parentSymbolTable.getNullValueSymbol(nullValueIdentifier);
	}
	
	public boolean isNullValue(String symbol) throws MaltChainedException {
		return parentSymbolTable.isNullValue(symbol);
	}
	
	public boolean isNullValue(int code) throws MaltChainedException {
		return parentSymbolTable.isNullValue(code);
	}
	
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		final ParseSymbolTable other = (ParseSymbolTable)obj;
		return ((name == null) ? other.name == null : name.equals(other.name));
	}

	public int hashCode() {
		return 217 + (null == name ? 0 : name.hashCode());
	}
	
	public String toString() {
		return parentSymbolTable.toString();
	}
}
